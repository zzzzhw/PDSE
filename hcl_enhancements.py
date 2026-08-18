from contextlib import contextmanager

import torch
import torch.nn.functional as F

from losses import HiDistanceLoss
from losses import HiDistanceXentLoss


EPS = 1e-12


def binary_focal_loss(probabilities, targets, gamma=2.0, alpha=0.25,
                      weight=None):
    """Binary focal loss for the HCL classifier's malware probability."""
    dtype_epsilon = torch.finfo(probabilities.dtype).eps
    probabilities = probabilities.clamp(
        min=dtype_epsilon,
        max=1.0 - dtype_epsilon,
    )
    targets = targets.float()
    p_t = targets * probabilities + (1.0 - targets) * (1.0 - probabilities)
    p_t = p_t.clamp(min=dtype_epsilon, max=1.0)
    alpha_t = targets * alpha + (1.0 - targets) * (1.0 - alpha)
    loss = -alpha_t * (1.0 - p_t).pow(gamma) * torch.log(p_t)
    if weight is not None:
        loss = loss * weight
    return loss.mean()


def mixup_batch(x_batch, y_binary_batch, alpha):
    """Mix HCL inputs and binary targets while retaining original family labels."""
    if alpha <= 0:
        raise ValueError('Mixup alpha must be positive')
    coefficient = float(torch.distributions.Beta(alpha, alpha).sample().item())
    permutation = torch.randperm(x_batch.shape[0], device=x_batch.device)
    mixed_x = coefficient * x_batch + (1.0 - coefficient) * x_batch[permutation]
    mixed_y = (
        coefficient * y_binary_batch
        + (1.0 - coefficient) * y_binary_batch[permutation]
    )
    return mixed_x, mixed_y, permutation, coefficient


def compute_hcl_training_loss(args, model, x_batch, y_batch,
                              y_binary_batch, weight_batch=None):
    """Compute HCL with an optional classification-side enhancement."""
    enhancement = getattr(args, 'hcl_enhancement', 'none')
    _, features, predictions = model(x_batch)

    if enhancement in ('none', 'sam', 'rwp', 'gaussian_noise', 'damp'):
        criterion = HiDistanceXentLoss().to(x_batch.device)
        return criterion(
            args.xent_lambda,
            predictions,
            y_binary_batch,
            features,
            labels=y_batch,
            margin=args.margin,
            weight=weight_batch,
        )

    distance_criterion = HiDistanceLoss().to(x_batch.device)
    distance_loss = distance_criterion(
        features,
        y_binary_batch,
        labels=y_batch,
        margin=args.margin,
        weight=None,
    )

    if enhancement == 'focal':
        classification_loss = binary_focal_loss(
            predictions[:, 1],
            y_binary_batch[:, 1],
            gamma=args.hcl_focal_gamma,
            alpha=args.hcl_focal_alpha,
            weight=weight_batch,
        )
    elif enhancement == 'mixup':
        mixed_x, mixed_y, permutation, coefficient = mixup_batch(
            x_batch,
            y_binary_batch,
            args.hcl_mixup_alpha,
        )
        _, _, mixed_predictions = model(mixed_x)
        mixed_weight = None
        if weight_batch is not None:
            mixed_weight = (
                coefficient * weight_batch
                + (1.0 - coefficient) * weight_batch[permutation]
            )
        classification_loss = F.binary_cross_entropy(
            mixed_predictions[:, 1],
            mixed_y[:, 1],
            reduction='mean',
            weight=mixed_weight,
        )
    else:
        raise ValueError(f'Unsupported HCL enhancement: {enhancement}')

    loss = distance_loss + args.xent_lambda * classification_loss
    return loss, distance_loss, classification_loss


@contextmanager
def sam_parameter_perturbation(model, rho):
    """Temporarily perturb only the HCL encoder toward SAM's worst case."""
    if not hasattr(model, 'encoder_model'):
        raise ValueError('HCL+SAM requires a model with an encoder_model module')
    parameters = [
        parameter
        for parameter in model.encoder_model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    if not parameters:
        yield
        return

    device = parameters[0].device
    grad_norm = torch.norm(torch.stack([
        parameter.grad.detach().norm(p=2).to(device)
        for parameter in parameters
    ]), p=2)
    if not torch.isfinite(grad_norm):
        raise FloatingPointError('SAM gradient norm is not finite')
    if grad_norm <= EPS:
        yield
        return
    scale = rho / (grad_norm + EPS)
    perturbations = []
    with torch.no_grad():
        for parameter in parameters:
            perturbation = parameter.grad.detach() * scale.to(parameter.device)
            parameter.add_(perturbation)
            perturbations.append((parameter, perturbation))

    try:
        yield
    finally:
        with torch.no_grad():
            for parameter, perturbation in perturbations:
                parameter.sub_(perturbation)


@contextmanager
def rwp_parameter_perturbation(model, gamma):
    """Temporarily apply filter-wise Gaussian noise to the HCL encoder."""
    if gamma <= 0:
        raise ValueError('RWP gamma must be positive')
    if not hasattr(model, 'encoder_model'):
        raise ValueError('HCL+RWP requires a model with an encoder_model module')

    perturbations = []
    for parameter in model.encoder_model.parameters():
        if not parameter.requires_grad:
            continue
        parameter_value = parameter.detach()
        if parameter.ndim > 1:
            filter_norms = parameter_value.reshape(
                parameter.shape[0], -1
            ).norm(p=2, dim=1)
            scale_shape = (parameter.shape[0],) + (1,) * (parameter.ndim - 1)
            noise_scale = gamma * filter_norms.reshape(scale_shape)
        else:
            noise_scale = gamma * (parameter_value.norm(p=2) + EPS)
        if not torch.isfinite(noise_scale).all():
            raise FloatingPointError('RWP parameter norm is not finite')
        perturbations.append((
            parameter,
            torch.randn_like(parameter) * noise_scale,
        ))

    with torch.no_grad():
        for parameter, perturbation in perturbations:
            parameter.add_(perturbation)

    try:
        yield
    finally:
        with torch.no_grad():
            for parameter, perturbation in perturbations:
                parameter.sub_(perturbation)


@contextmanager
def gaussian_weight_perturbation(model, std):
    """Temporarily add fixed-scale Gaussian noise to the HCL encoder."""
    if std <= 0:
        raise ValueError('Gaussian weight noise std must be positive')
    if not hasattr(model, 'encoder_model'):
        raise ValueError(
            'HCL+Gaussian noise requires a model with an encoder_model module'
        )

    perturbations = [
        (parameter, torch.randn_like(parameter) * std)
        for parameter in model.encoder_model.parameters()
        if parameter.requires_grad
    ]
    with torch.no_grad():
        for parameter, perturbation in perturbations:
            parameter.add_(perturbation)

    try:
        yield
    finally:
        with torch.no_grad():
            for parameter, perturbation in perturbations:
                parameter.sub_(perturbation)


@contextmanager
def damp_parameter_perturbation(model, std):
    """Temporarily apply element-wise multiplicative Gaussian perturbations."""
    if std <= 0:
        raise ValueError('DAMP std must be positive')
    if not hasattr(model, 'encoder_model'):
        raise ValueError('HCL+DAMP requires a model with an encoder_model module')

    perturbations = []
    for name, parameter in model.encoder_model.named_parameters():
        if not parameter.requires_grad or not name.endswith('weight'):
            continue
        multiplier = 1.0 + torch.randn_like(parameter) * std
        perturbation = parameter.detach() * (multiplier - 1.0)
        perturbations.append((parameter, perturbation, multiplier))

    with torch.no_grad():
        for parameter, perturbation, _ in perturbations:
            parameter.add_(perturbation)

    try:
        yield
    finally:
        with torch.no_grad():
            for parameter, perturbation, multiplier in perturbations:
                parameter.sub_(perturbation)
                if parameter.grad is not None:
                    parameter.grad.mul_(multiplier)
