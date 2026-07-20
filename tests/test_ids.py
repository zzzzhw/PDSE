import unittest
from types import SimpleNamespace

import numpy as np
import torch

from ids import CADEIDS
from ids import HCLIDS
from ids import add_weight_diff
from ids import build_exposure_indices
from ids import build_exposure_loader
from model import CAE
from model import SimpleEncClassifier


class IDSTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        np.random.seed(7)
        self.X = np.random.randn(12, 4).astype(np.float32)
        self.y_family = np.array([0, 0, 0, 1, 1, 2, 2, 3, 0, 4, 2, 5])
        self.y_binary = (self.y_family != 0).astype(np.int64)
        self.args = SimpleNamespace(
            ids_proxy_lr=0.1,
            ids_ema_decay=0.6,
            ids_lambda=1e-2,
            ids_gamma=1.0,
            ids_robust_weight=0.5,
            ids_grad_clip=1.0,
            xent_lambda=1.0,
            cae_lambda=0.1,
            margin=1.0,
        )

    def test_exposure_triplets_preserve_label_roles(self):
        indices = build_exposure_indices(
            self.y_family,
            self.y_binary,
            exposure_count=3,
            seed=11,
            balance_binary=False,
        )
        history_end = len(self.y_family) - 3
        self.assertEqual(indices.shape, (9,))
        for current, similar, opposite in indices.reshape(-1, 3):
            self.assertGreaterEqual(current, history_end)
            self.assertLess(similar, history_end)
            self.assertLess(opposite, history_end)
            self.assertEqual(self.y_binary[current], self.y_binary[similar])
            self.assertNotEqual(self.y_binary[current], self.y_binary[opposite])

    def test_exposure_balances_current_binary_labels(self):
        y_family = self.y_family.copy()
        y_family[-1] = 0
        y_binary = (y_family != 0).astype(np.int64)
        indices = build_exposure_indices(
            y_family, y_binary, exposure_count=3, seed=11, balance_binary=True
        )
        current = indices.reshape(-1, 3)[:, 0]
        counts = np.bincount(y_binary[current], minlength=2)
        self.assertEqual(counts[0], counts[1])

    def test_exposure_loader_emits_role_major_triplets(self):
        X = np.arange(len(self.y_family), dtype=np.float32).reshape(-1, 1)
        loader, indices = build_exposure_loader(
            X,
            self.y_family,
            self.y_binary,
            exposure_count=3,
            batch_size=9,
            seed=11,
            balance_binary=False,
        )
        x_batch, y_batch, y_binary_batch = next(iter(loader))
        expected_indices = indices.reshape(-1, 3).T.reshape(-1)

        np.testing.assert_array_equal(x_batch[:, 0].numpy(), expected_indices)
        np.testing.assert_array_equal(
            y_batch.numpy(), self.y_family[expected_indices]
        )
        np.testing.assert_array_equal(
            y_binary_batch.argmax(dim=1).numpy(), self.y_binary[expected_indices]
        )

    def test_ids_search_restore_and_robust_update(self):
        model = SimpleEncClassifier([4, 8, 4], [4, 4, 2], dropout=0, verbose=0)
        model(torch.from_numpy(self.X[:2]))
        loader, _ = build_exposure_loader(
            self.X,
            self.y_family,
            self.y_binary,
            exposure_count=3,
            batch_size=9,
            seed=11,
        )
        x_batch, y_batch, y_binary_batch = next(iter(loader))
        controller = HCLIDS(model, self.args, torch.device('cpu'))
        search = controller.update(
            model, x_batch, y_batch, y_binary_batch, self.args
        )

        self.assertTrue(controller.diff)
        self.assertTrue(all(name.startswith('encoder_model.') for name in controller.diff))
        self.assertTrue(np.isfinite(search['search_loss']))
        self.assertTrue(np.isfinite(search['feature_distance']))

        before_perturb = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
        }
        add_weight_diff(model, controller.diff, self.args.ids_lambda)
        add_weight_diff(model, controller.diff, -self.args.ids_lambda)
        for name, parameter in model.named_parameters():
            self.assertTrue(torch.allclose(parameter, before_perturb[name], atol=1e-7))

        before_update = {
            name: parameter.detach().clone()
            for name, parameter in model.encoder_model.named_parameters()
        }
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        robust_loss = controller.robust_step(
            model, optimizer, x_batch, y_batch, y_binary_batch, self.args
        )
        self.assertTrue(np.isfinite(robust_loss))
        changed = any(
            not torch.allclose(parameter, before_update[name])
            for name, parameter in model.encoder_model.named_parameters()
        )
        self.assertTrue(changed)

    def test_cade_ids_search_restore_and_robust_update(self):
        model = CAE([4, 8, 3], verbose=0)
        loader, _ = build_exposure_loader(
            self.X,
            self.y_family,
            self.y_binary,
            exposure_count=3,
            batch_size=9,
            seed=11,
            balance_binary=False,
        )
        x_batch, y_batch, y_binary_batch = next(iter(loader))
        controller = CADEIDS(model, self.args, torch.device('cpu'))
        before_search = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
        }
        search = controller.update(
            model, x_batch, y_batch, y_binary_batch, self.args
        )

        self.assertTrue(controller.diff)
        self.assertTrue(all(name.startswith('encoder_model.') for name in controller.diff))
        self.assertTrue(np.isfinite(search['search_loss']))
        self.assertTrue(np.isfinite(search['feature_distance']))
        for name, parameter in model.named_parameters():
            self.assertTrue(torch.equal(parameter, before_search[name]))

        before_update = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
        }
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        robust_loss = controller.robust_step(
            model, optimizer, x_batch, y_batch, y_binary_batch, self.args
        )
        self.assertTrue(np.isfinite(robust_loss))
        self.assertTrue(any(
            not torch.allclose(parameter, before_update[name])
            for name, parameter in model.named_parameters()
        ))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA integration test')
    def test_train_encoder_uses_ids_exposure(self):
        from train import train_encoder

        model = SimpleEncClassifier([4, 8, 4], [4, 4, 2], dropout=0, verbose=0)
        before_update = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
        }
        args = SimpleNamespace(
            sampler='random',
            bsize=6,
            loss_func='hi-dist-xent',
            xent_lambda=1.0,
            margin=1.0,
            display_interval=1000,
            encoder='simple-enc-mlp',
            ids=True,
            ids_proxy_lr=0.1,
            ids_ema_decay=0.6,
            ids_lambda=1e-2,
            ids_gamma=1.0,
            ids_robust_weight=0.5,
            ids_grad_clip=1.0,
            ids_batch_size=9,
            ids_unbalanced_exposure=False,
            ids_warmup=0,
            seed=11,
            snapshot=False,
            epochs=1,
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        train_encoder(
            args,
            model,
            self.X,
            self.y_family,
            self.y_binary,
            optimizer,
            total_epochs=1,
            model_path='unused.pth',
            adjust=False,
            ids_exposure_count=3,
        )
        changed = any(
            not torch.allclose(parameter.cpu(), before_update[name])
            for name, parameter in model.named_parameters()
        )
        self.assertTrue(changed)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA integration test')
    def test_train_cade_encoder_uses_ids_exposure(self):
        from train import train_encoder

        model = CAE([4, 8, 3], verbose=0)
        before_update = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
        }
        args = SimpleNamespace(
            sampler='triplet',
            bsize=9,
            loss_func='triplet-mse',
            cae_lambda=0.1,
            margin=1.0,
            display_interval=1000,
            encoder='cae',
            ids=True,
            ids_proxy_lr=0.1,
            ids_ema_decay=0.6,
            ids_lambda=1e-2,
            ids_gamma=1.0,
            ids_robust_weight=0.5,
            ids_grad_clip=1.0,
            ids_batch_size=9,
            ids_unbalanced_exposure=False,
            ids_warmup=0,
            seed=11,
            snapshot=False,
            epochs=1,
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        train_encoder(
            args,
            model,
            self.X,
            self.y_family,
            self.y_binary,
            optimizer,
            total_epochs=1,
            model_path='unused.pth',
            adjust=False,
            ids_exposure_count=3,
        )
        changed = any(
            not torch.allclose(parameter.cpu(), before_update[name])
            for name, parameter in model.named_parameters()
        )
        self.assertTrue(changed)


if __name__ == '__main__':
    unittest.main()
