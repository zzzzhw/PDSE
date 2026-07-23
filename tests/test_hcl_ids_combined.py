import unittest
from argparse import Namespace
from collections import OrderedDict
import sys
import types
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, TensorDataset

from ids import HCLIDS
from model import SimpleEncClassifier

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


class HCLIDSCombinedTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.args = Namespace(
            ids_proxy_lr=0.1,
            ids_ema_decay=0.0,
            ids_lambda=0.01,
            ids_gamma=1.0,
            ids_robust_weight=0.5,
            ids_grad_clip=10.0,
            ids_hcl_update_mode='combined',
            xent_lambda=1.0,
            margin=1.0,
            ids_warmup=0,
            loss_func='hi-dist-xent',
            display_interval=100,
        )
        self.model = SimpleEncClassifier([4, 3], [3, 2], dropout=0.0, verbose=0)
        self.controller = HCLIDS(self.model, self.args, torch.device('cpu'))
        self.controller.diff = OrderedDict(
            (name, torch.full_like(parameter, 0.25))
            for name, parameter in self.model.named_parameters()
            if name.startswith('encoder_model.') and name.endswith('weight')
        )
        self.x = torch.tensor([
            [1.0, 0.0, 0.5, 0.0],
            [0.8, 0.1, 0.4, 0.2],
            [0.0, 1.0, 0.0, 0.5],
            [0.1, 0.9, 0.2, 0.4],
        ])
        self.y = torch.tensor([0, 0, 1, 1])
        self.y_binary = torch.tensor([
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ])

    def test_combined_update_restores_perturbation_and_steps_once(self):
        optimizer = CountingSGD(self.model.parameters(), lr=0.01)
        initial = {
            name: parameter.detach().clone()
            for name, parameter in self.model.named_parameters()
        }

        optimizer.zero_grad()
        normal_loss, _, _, _ = self.controller._loss(
            self.model, self.x, self.y, self.y_binary, self.args
        )
        normal_loss.backward()
        normal_grads = {
            name: parameter.grad.detach().clone()
            for name, parameter in self.model.named_parameters()
        }

        robust_loss = self.controller.accumulate_robust_grad(
            self.model, self.x, self.y, self.y_binary, self.args
        )

        self.assertGreater(robust_loss, 0.0)
        self.assertEqual(optimizer.step_count, 0)
        self.assertTrue(any(
            not torch.equal(parameter.grad, normal_grads[name])
            for name, parameter in self.model.named_parameters()
        ))
        for name, parameter in self.model.named_parameters():
            self.assertTrue(torch.equal(parameter, initial[name]), name)

        optimizer.step()
        self.assertEqual(optimizer.step_count, 1)
        self.assertTrue(any(
            not torch.equal(parameter, initial[name])
            for name, parameter in self.model.named_parameters()
        ))

    def test_training_loop_uses_one_combined_optimizer_step(self):
        dataset = TensorDataset(
            self.x,
            self.y,
            self.y_binary,
            torch.ones(self.x.shape[0]),
        )
        train_loader = DataLoader(dataset, batch_size=self.x.shape[0])
        ids_loader = [(self.x, self.y, self.y_binary)]
        optimizer = CountingSGD(self.model.parameters(), lr=0.01)

        with patch.object(torch.nn.Module, 'cuda', lambda module: module):
            loss = train_encoder_one_epoch(
                self.args,
                self.model,
                train_loader,
                optimizer,
                epoch=1,
                ids_loader=ids_loader,
                ids_controller=self.controller,
            )

        self.assertGreater(loss, 0.0)
        self.assertEqual(optimizer.step_count, 1)


if __name__ == '__main__':
    unittest.main()
