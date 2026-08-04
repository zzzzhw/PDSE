import unittest
from argparse import Namespace
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, TensorDataset

from model import SimpleEncClassifier
from pdse import HCLPDSE
from train import train_encoder_one_epoch


class CountingSGD(torch.optim.SGD):
    def __init__(self, params, **kwargs):
        super().__init__(params, **kwargs)
        self.step_count = 0

    def step(self, closure=None):
        self.step_count += 1
        return super().step(closure)


class HCLPDSEAblationTest(unittest.TestCase):
    def test_no_perturbation_uses_clean_exposure_loss_and_one_update(self):
        torch.manual_seed(17)
        args = Namespace(
            pdse_proxy_lr=0.1,
            pdse_ema_decay=0.0,
            pdse_lambda=0.01,
            pdse_gamma=1.0,
            pdse_robust_weight=0.5,
            pdse_grad_clip=10.0,
            pdse_hcl_update_mode='combined',
            pdse_ablation='no-perturbation',
            pdse_warmup=0,
            xent_lambda=1.0,
            margin=1.0,
            loss_func='hi-dist-xent',
            display_interval=100,
            hcl_enhancement='none',
        )
        model = SimpleEncClassifier([4, 3], [3, 2], dropout=0.0, verbose=0)
        controller = HCLPDSE(model, args, torch.device('cpu'))
        x = torch.tensor([
            [1.0, 0.0, 0.5, 0.0],
            [0.8, 0.1, 0.4, 0.2],
            [0.0, 1.0, 0.0, 0.5],
            [0.1, 0.9, 0.2, 0.4],
        ])
        y = torch.tensor([0, 0, 1, 1])
        y_binary = torch.tensor([
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ])
        train_loader = DataLoader(
            TensorDataset(x, y, y_binary, torch.ones(x.shape[0])),
            batch_size=x.shape[0],
        )
        optimizer = CountingSGD(model.parameters(), lr=0.01)

        with patch.object(controller, 'update', wraps=controller.update) as search:
            with patch.object(
                controller,
                'accumulate_robust_grad',
                wraps=controller.accumulate_robust_grad,
            ) as perturbed:
                with patch.object(
                    controller,
                    'accumulate_clean_exposure_grad',
                    wraps=controller.accumulate_clean_exposure_grad,
                ) as clean:
                    loss = train_encoder_one_epoch(
                        args,
                        model,
                        train_loader,
                        optimizer,
                        epoch=1,
                        pdse_loader=[(x, y, y_binary)],
                        pdse_controller=controller,
                    )

        self.assertGreater(loss, 0.0)
        self.assertEqual(search.call_count, 0)
        self.assertEqual(perturbed.call_count, 0)
        self.assertEqual(clean.call_count, 1)
        self.assertEqual(optimizer.step_count, 1)


if __name__ == '__main__':
    unittest.main()
