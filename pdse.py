from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from common import to_categorical
from losses import HiDistanceXentLoss
from losses import TripletMSELoss


EPS = 1e-20


def _pdse_option(args, name):
    """Read the PDSE option while accepting pre-rename Namespace objects."""
    pdse_name = f'pdse_{name}'
    if hasattr(args, pdse_name):
        return getattr(args, pdse_name)
    return getattr(args, f'ids_{name}')


def normalized_weight_diff(model, proxy, parameter_prefix='encoder_model.'):
    """Return the scale-normalized proxy update used by PDSE."""
    proxy_params = dict(proxy.named_parameters())
    diff = OrderedDict()
    for name, weight in model.named_parameters():
        if not name.startswith(parameter_prefix) or not name.endswith('weight'):
            continue
        proxy_weight = proxy_params[name]
        delta = proxy_weight.detach() - weight.detach()
        diff[name] = (weight.detach().norm() / (delta.norm() + EPS) * delta).clone()
    return diff


def add_weight_diff(model, diff, coefficient=1.0):
    """Add a stored perturbation to model parameters in place."""
    if not diff:
        return
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name in diff:
                parameter.add_(coefficient * diff[name].to(parameter.device))


def average_weight_diff(current, new, decay):
    """Update the perturbation with an exponential moving average."""
    if current is None:
        return OrderedDict((name, value.clone()) for name, value in new.items())
    for name, value in new.items():
        if name not in current:
            current[name] = value.clone()
        else:
            current[name].mul_(decay).add_(value, alpha=1.0 - decay)
    return current


def _month_ordinals(timestamps, sample_count):
    """Convert sample timestamps to comparable calendar-month ordinals."""
    timestamps = np.asarray(timestamps)
    if timestamps.shape != (sample_count,):
        raise ValueError('PDSE timestamps must contain one value per training sample')
    try:
        months = timestamps.astype('datetime64[M]')
    except (TypeError, ValueError) as error:
        raise ValueError('PDSE timestamps must be valid date-like values') from error
    if np.isnat(months).any():
        raise ValueError('PDSE timestamps cannot contain missing dates')
    return months.astype(np.int64)


def _nearest_timestamp_candidates(candidates, current_index, month_ordinals):
    distances = np.abs(month_ordinals[candidates] - month_ordinals[current_index])
    return candidates[distances == distances.min()]


def build_exposure_indices(y_family, y_binary, exposure_count, timestamps, seed=0):
    """Build one time-local drift triplet for every newly labeled sample."""
    y_family = np.asarray(y_family)
    y_binary = np.asarray(y_binary)
    sample_count = y_family.shape[0]
    if y_binary.shape != (sample_count,):
        raise ValueError('PDSE binary labels must align with family labels')
    if exposure_count <= 0 or exposure_count >= sample_count:
        raise ValueError('exposure_count must identify a non-empty tail of the training set')
    month_ordinals = _month_ordinals(timestamps, sample_count)

    history_end = sample_count - exposure_count
    history_indices = np.arange(history_end)
    current_indices = np.arange(history_end, sample_count)
    rng = np.random.default_rng(seed)
    triplets = []

    for current_index in current_indices:
        same_candidates = history_indices[y_family[:history_end] == y_family[current_index]]
        if same_candidates.size == 0:
            same_candidates = history_indices[y_binary[:history_end] == y_binary[current_index]]
        opposite_candidates = history_indices[y_binary[:history_end] != y_binary[current_index]]
        if same_candidates.size == 0 or opposite_candidates.size == 0:
            raise ValueError('PDSE exposure construction requires both benign and malicious history')

        nearest_same = _nearest_timestamp_candidates(
            same_candidates, current_index, month_ordinals
        )
        nearest_opposite = _nearest_timestamp_candidates(
            opposite_candidates, current_index, month_ordinals
        )
        same_index = int(rng.choice(nearest_same))
        opposite_index = int(rng.choice(nearest_opposite))
        triplets.append((int(current_index), same_index, opposite_index))

    rng.shuffle(triplets)
    return np.asarray(triplets, dtype=np.int64).reshape(-1)


def build_exposure_loader(X, y_family, y_binary, exposure_count, timestamps,
                          batch_size, seed=0):
    """Create role-major drift-triplet batches centered on newly labeled samples."""
    indices = build_exposure_indices(
        y_family,
        y_binary,
        exposure_count,
        timestamps,
        seed=seed,
    )
    triplet_indices = indices.reshape(-1, 3)
    triplet_batch_size = max(1, min(int(batch_size) // 3, triplet_indices.shape[0]))

    X_tensor = torch.from_numpy(np.asarray(X)[triplet_indices]).float()
    y_tensor = torch.from_numpy(np.asarray(y_family)[triplet_indices]).long()
    y_binary_tensor = torch.from_numpy(to_categorical(
        np.asarray(y_binary)[triplet_indices], num_classes=2
    ).reshape(triplet_indices.shape[0], 3, 2)).float()
    dataset = TensorDataset(X_tensor, y_tensor, y_binary_tensor)
    loader = DataLoader(
        dataset,
        batch_size=triplet_batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=_collate_role_major_triplets,
    )
    return loader, indices


def _collate_role_major_triplets(batch):
    """Flatten triplets as all anchors, all positives, then all negatives."""
    X_batch, y_batch, y_binary_batch = zip(*batch)
    X_batch = torch.stack(X_batch).transpose(0, 1).flatten(0, 1)
    y_batch = torch.stack(y_batch).transpose(0, 1).flatten()
    y_binary_batch = torch.stack(y_binary_batch).transpose(0, 1).flatten(0, 1)
    return X_batch, y_batch, y_binary_batch


class HCLPDSE:
    """First-order PDSE adaptation for the hierarchical contrastive classifier."""

    def __init__(self, model, args, device):
        if not hasattr(model, 'encoder_model'):
            raise ValueError('HCL+PDSE requires a model with an encoder_model module')
        self.device = device
        dropout = next(
            (module.p for module in model.mlp_model.modules() if isinstance(module, nn.Dropout)),
            0.0,
        )
        self.proxy = type(model)(
            model.enc_dims,
            model.mlp_dims,
            dropout=dropout,
            verbose=0,
        ).to(device)
        self.proxy.load_state_dict(model.state_dict())
        self.proxy_optimizer = torch.optim.SGD(
            self.proxy.encoder_model.parameters(), lr=_pdse_option(args, 'proxy_lr')
        )
        self.ema_decay = _pdse_option(args, 'ema_decay')
        self.perturb_scale = _pdse_option(args, 'lambda')
        self.constraint_weight = _pdse_option(args, 'gamma')
        self.robust_weight = _pdse_option(args, 'robust_weight')
        self.grad_clip = _pdse_option(args, 'grad_clip')
        self.update_mode = _pdse_option(args, 'hcl_update_mode')
        self.diff = None

    @staticmethod
    def _loss(model, x_batch, y_batch, y_binary_batch, args):
        _, features, predictions = model(x_batch)
        criterion = HiDistanceXentLoss().to(x_batch.device)
        loss, distance_loss, xent_loss = criterion(
            args.xent_lambda,
            predictions,
            y_binary_batch,
            features,
            labels=y_batch,
            margin=args.margin,
        )
        return loss, distance_loss, xent_loss, features

    def update(self, model, x_batch, y_batch, y_binary_batch, args):
        """Search one bounded, high-HCL-loss encoder perturbation."""
        self.proxy.load_state_dict(model.state_dict())
        self.proxy.train()

        with torch.no_grad():
            reference_features = F.normalize(model.encode(x_batch), dim=1)
        if self.diff:
            add_weight_diff(self.proxy, self.diff, self.perturb_scale)

        self.proxy_optimizer.zero_grad()
        hcl_loss, _, _, proxy_features = self._loss(
            self.proxy, x_batch, y_batch, y_binary_batch, args
        )
        feature_distance = F.mse_loss(
            F.normalize(proxy_features, dim=1), reference_features
        )
        adversary_loss = -hcl_loss + self.constraint_weight * feature_distance
        adversary_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.proxy.encoder_model.parameters(), self.grad_clip)
        self.proxy_optimizer.step()

        new_diff = normalized_weight_diff(model, self.proxy)
        self.diff = average_weight_diff(self.diff, new_diff, self.ema_decay)
        return {
            'search_loss': float(hcl_loss.detach()),
            'feature_distance': float(feature_distance.detach()),
        }

    def robust_step(self, model, optimizer, x_batch, y_batch, y_binary_batch, args):
        """Apply the legacy standalone HCL robust optimizer step."""
        if not self.diff:
            return 0.0

        add_weight_diff(model, self.diff, self.perturb_scale)
        try:
            optimizer.zero_grad()
            robust_loss, _, _, _ = self._loss(
                model, x_batch, y_batch, y_binary_batch, args
            )
            (self.robust_weight * robust_loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
            optimizer.step()
        finally:
            add_weight_diff(model, self.diff, -self.perturb_scale)
        return float(robust_loss.detach())

    def accumulate_robust_grad(self, model, x_batch, y_batch, y_binary_batch, args):
        """Accumulate perturbed HCL gradients without updating model parameters."""
        if not self.diff:
            return 0.0

        add_weight_diff(model, self.diff, self.perturb_scale)
        try:
            robust_loss, _, _, _ = self._loss(
                model, x_batch, y_batch, y_binary_batch, args
            )
            (self.robust_weight * robust_loss).backward()
        finally:
            add_weight_diff(model, self.diff, -self.perturb_scale)
        return float(robust_loss.detach())


class CADEPDSE:
    """First-order PDSE adaptation for CADE's triplet autoencoder."""

    def __init__(self, model, args, device):
        if not hasattr(model, 'encoder_model') or not hasattr(model, 'decoder_model'):
            raise ValueError('CADE+PDSE requires a model with encoder and decoder modules')
        self.device = device
        self.proxy = type(model)(model.enc_dims, verbose=0).to(device)
        self.proxy.load_state_dict(model.state_dict())
        self.proxy_optimizer = torch.optim.SGD(
            self.proxy.encoder_model.parameters(), lr=_pdse_option(args, 'proxy_lr')
        )
        self.ema_decay = _pdse_option(args, 'ema_decay')
        self.perturb_scale = _pdse_option(args, 'lambda')
        self.constraint_weight = _pdse_option(args, 'gamma')
        self.robust_weight = _pdse_option(args, 'robust_weight')
        self.grad_clip = _pdse_option(args, 'grad_clip')
        self.update_mode = _pdse_option(args, 'cade_update_mode')
        self.diff = None

    @staticmethod
    def _loss(model, x_batch, y_batch, y_binary_batch, args):
        del y_binary_batch
        features, decoded = model(x_batch)
        criterion = TripletMSELoss().to(x_batch.device)
        loss, triplet_loss, mse_loss = criterion(
            args.cae_lambda,
            x_batch,
            decoded,
            features,
            labels=y_batch,
            margin=args.margin,
        )
        return loss, triplet_loss, mse_loss, features

    def update(self, model, x_batch, y_batch, y_binary_batch, args):
        """Search one bounded, high-CADE-loss encoder perturbation."""
        self.proxy.load_state_dict(model.state_dict())
        self.proxy.train()

        with torch.no_grad():
            reference_features = F.normalize(model.encode(x_batch), dim=1)
        if self.diff:
            add_weight_diff(self.proxy, self.diff, self.perturb_scale)

        self.proxy_optimizer.zero_grad()
        cade_loss, _, _, proxy_features = self._loss(
            self.proxy, x_batch, y_batch, y_binary_batch, args
        )
        feature_distance = F.mse_loss(
            F.normalize(proxy_features, dim=1), reference_features
        )
        adversary_loss = -cade_loss + self.constraint_weight * feature_distance
        adversary_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.proxy.encoder_model.parameters(), self.grad_clip
        )
        self.proxy_optimizer.step()

        new_diff = normalized_weight_diff(model, self.proxy)
        self.diff = average_weight_diff(self.diff, new_diff, self.ema_decay)
        return {
            'search_loss': float(cade_loss.detach()),
            'feature_distance': float(feature_distance.detach()),
        }

    def robust_step(self, model, optimizer, x_batch, y_batch, y_binary_batch, args):
        """Apply the legacy standalone CADE robust optimizer step."""
        if not self.diff:
            return 0.0

        add_weight_diff(model, self.diff, self.perturb_scale)
        try:
            optimizer.zero_grad()
            robust_loss, _, _, _ = self._loss(
                model, x_batch, y_batch, y_binary_batch, args
            )
            (self.robust_weight * robust_loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
            optimizer.step()
        finally:
            add_weight_diff(model, self.diff, -self.perturb_scale)
        return float(robust_loss.detach())

    def accumulate_robust_grad(self, model, x_batch, y_batch, y_binary_batch, args):
        """Accumulate perturbed CADE gradients without updating model parameters."""
        if not self.diff:
            return 0.0

        add_weight_diff(model, self.diff, self.perturb_scale)
        try:
            robust_loss, _, _, _ = self._loss(
                model, x_batch, y_batch, y_binary_batch, args
            )
            (self.robust_weight * robust_loss).backward()
        finally:
            add_weight_diff(model, self.diff, -self.perturb_scale)
        return float(robust_loss.detach())


# Compatibility aliases for experiments created before the PDSE rename.
HCLIDS = HCLPDSE
CADEIDS = CADEPDSE
