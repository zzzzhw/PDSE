import sys
import types
import unittest
from argparse import Namespace
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from model import SimpleEncClassifier
from hcl_enhancements import binary_focal_loss

try:
    import pytorch_metric_learning.samplers  # noqa: F401
except ModuleNotFoundError:
    metric_learning = types.ModuleType('pytorch_metric_learning')
    metric_samplers = types.ModuleType('pytorch_metric_learning.samplers')
    metric_samplers.MPerClassSampler = object
    metric_learning.samplers = metric_samplers
    sys.modules.setdefault('pytorch_metric_learning', metric_learning)
    sys.modules.setdefault('pytorch_metric_learning.samplers', metric_samplers)
    project_samplers = types.ModuleType('samplers')
    project_samplers.ProportionalClassSampler = object
    project_samplers.HalfSampler = object
    project_samplers.TripletSampler = object
    sys.modules.setdefault('samplers', project_samplers)

from train import train_encoder_one_epoch


class CountingSGD(torch.optim.SGD):
    def __init__(self, params, **kwargs):
        super().__init__(params, **kwargs)
        self.step_count = 0

    def step(self, closure=None):
        self.step_count += 1
        return super().step(closure)


class HCLTrainingEnhancementTest(unittest.TestCase):
    def setUp(self):
        self.x = torch.tensor([
            [1.0, 0.0, 0.4, 0.0],
            [0.8, 0.1, 0.5, 0.1],
            [0.0, 1.0, 0.0, 0.5],
            [0.1, 0.9, 0.2, 0.4],
        ])
        self.y_family = torch.tensor([0, 0, 1, 1])
        self.y_binary = torch.tensor([
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ])
        self.loader = DataLoader(
            TensorDataset(
                self.x,
                self.y_family,
                self.y_binary,
                torch.ones(self.x.shape[0]),
            ),
            batch_size=self.x.shape[0],
        )

    def _run_one_batch(self, enhancement):
        torch.manual_seed(11)
        args = Namespace(
            hcl_enhancement=enhancement,
            hcl_mixup_alpha=0.2,
            hcl_focal_gamma=2.0,
            hcl_focal_alpha=0.25,
            hcl_sam_rho=0.05,
            xent_lambda=1.0,
            margin=1.0,
            loss_func='hi-dist-xent',
            display_interval=100,
            pdse_warmup=0,
        )
        model = SimpleEncClassifier([4, 3], [3, 2], dropout=0.0, verbose=0)
        optimizer = CountingSGD(model.parameters(), lr=0.01)
        initial = [parameter.detach().clone() for parameter in model.parameters()]

        with patch.object(torch.cuda, 'is_available', return_value=False), \
                patch.object(torch.nn.Module, 'cuda', lambda module: module):
            loss = train_encoder_one_epoch(
                args,
                model,
                self.loader,
                optimizer,
                epoch=1,
            )

        self.assertTrue(torch.isfinite(torch.tensor(loss)), enhancement)
        self.assertGreater(loss, 0.0, enhancement)
        self.assertEqual(optimizer.step_count, 1, enhancement)
        self.assertTrue(any(
            not torch.equal(parameter, before)
            for parameter, before in zip(model.parameters(), initial)
        ), enhancement)

    def test_all_enhancement_modes_complete_one_training_batch(self):
        for enhancement in ('none', 'mixup', 'focal', 'sam'):
            with self.subTest(enhancement=enhancement):
                self._run_one_batch(enhancement)

    def test_focal_loss_is_finite_for_saturated_wrong_predictions(self):
        probabilities = torch.tensor([1.0, 0.0], requires_grad=True)
        targets = torch.tensor([0.0, 1.0])

        loss = binary_focal_loss(probabilities, targets)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(probabilities.grad).all())


if __name__ == '__main__':
    unittest.main()
