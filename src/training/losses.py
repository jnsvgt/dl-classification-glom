"""Loss functions for imbalanced classification."""

import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LDAM — Label-Distribution-Aware Margin Loss (Cao et al., NeurIPS 2019)
# ---------------------------------------------------------------------------

class NormedLinear(nn.Module):
    """Linear layer with L2-normalised features and weights (needed by LDAM)."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        return F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))


class LDAMLoss(nn.Module):
    r"""Per-class margins inversely proportional to :math:`n_j^{1/4}`.

    Combined with *Deferred Re-Weighting* (DRW): class-balanced weights are
    applied only after the model has learnt initial representations.
    """

    def __init__(self, cls_num_list, max_m=0.5, s=30.0, weight=None):
        super().__init__()
        arr = cls_num_list.cpu().numpy() if isinstance(cls_num_list, torch.Tensor) else np.array(cls_num_list)
        m = 1.0 / np.sqrt(np.sqrt(arr))
        m = m * (max_m / m.max())
        self.register_buffer("m_list", torch.FloatTensor(m))
        self.s = s
        self.weight = weight

    def update_weight(self, weight):
        """Called by the trainer when DRW kicks in."""
        self.weight = weight

    def forward(self, x, target):
        idx = torch.zeros_like(x, dtype=torch.bool)
        idx.scatter_(1, target.view(-1, 1), True)
        batch_m = self.m_list[None, :].mm(idx.float().t()).view(-1, 1)
        x_m = x - batch_m
        output = torch.where(idx, x_m, x)
        return F.cross_entropy(self.s * output, target, weight=self.weight)


def compute_drw_weights(cls_num_list, beta=0.9999, device="cuda"):
    arr = cls_num_list.cpu().numpy() if isinstance(cls_num_list, torch.Tensor) else np.array(cls_num_list)
    eff = 1.0 - np.power(beta, arr)
    w = (1.0 - beta) / eff
    w = w / w.sum() * len(arr)
    return torch.FloatTensor(w).to(device)


def wrap_model_with_ldam(model, model_name, n_classes):
    """Replace the classifier head with a NormedLinear layer."""
    if model_name == "resnet50":
        n = model.fc.in_features
        model.fc = NormedLinear(n, n_classes)
    elif model_name in ("swinv2_tiny", "convnext_tiny", "vit_large"):
        if hasattr(model, "head"):
            if hasattr(model.head, "fc"):
                n = model.head.fc.in_features
                model.head.fc = NormedLinear(n, n_classes)
            elif isinstance(model.head, nn.Linear):
                n = model.head.in_features
                model.head = NormedLinear(n, n_classes)
            else:
                n = model.num_features
                model.head = NormedLinear(n, n_classes)
    elif model_name == "phikon_v2":
        n = model.classifier.in_features
        model.classifier = NormedLinear(n, n_classes)
    else:
        raise ValueError(f"LDAM wrapping not implemented for {model_name}")
    logger.info(f"Replaced classifier with NormedLinear for LDAM ({model_name})")
    return model


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_loss(optimization, class_weights=None, class_counts=None,
                label_smoothing=0.05, ldam_max_m=0.5, ldam_scale=30.0,
                num_classes=None, max_margin=None, **_kw):
    """Create a loss function based on the optimisation strategy name."""
    if max_margin is not None:
        ldam_max_m = max_margin

    if optimization in ("baseline", "weighted_sampler"):
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    if optimization == "weighted_ce":
        if class_weights is None and class_counts is not None:
            # inverse-frequency weighting
            counts = torch.tensor(class_counts, dtype=torch.float)
            class_weights = 1.0 / (counts + 1e-6)
            class_weights = class_weights / class_weights.sum() * len(class_weights)
        if class_weights is None:
            raise ValueError("class_weights or class_counts required for weighted_ce")
        if not isinstance(class_weights, torch.Tensor):
            class_weights = torch.tensor(class_weights, dtype=torch.float)
        return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)

    if optimization == "ldam":
        if class_counts is None:
            raise ValueError("class_counts required for LDAM")
        logger.info(f"LDAM loss: max_m={ldam_max_m}, scale={ldam_scale}")
        return LDAMLoss(class_counts, max_m=ldam_max_m, s=ldam_scale)

    raise ValueError(
        f"Unknown optimisation: '{optimization}'. "
        "Choose from: baseline, weighted_sampler, weighted_ce, ldam"
    )
