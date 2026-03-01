"""Model factory — creates classification models from timm / HuggingFace."""

import timm
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights

from src.models.phikon import create_phikon_v2

# The five architectures we evaluate:
#   resnet50       – CNN baseline (25.6M params)
#   convnext_tiny  – modern CNN (28.6M)
#   swinv2_tiny    – sliding-window ViT (28.3M)
#   vit_large      – vanilla ViT-L/16, IN-21k pre-trained (304M)
#   phikon_v2      – Owkin histopathology ViT-L/16 (304M)
SUPPORTED_MODELS = {
    "resnet50",
    "convnext_tiny",
    "swinv2_tiny",
    "vit_large",
    "phikon_v2",
}

# short aliases accepted by CLI
_ALIASES = {
    "convnext": "convnext_tiny",
    "swinv2": "swinv2_tiny",
}

# Transformer models whose CAM output needs spatial reshaping
_RESHAPE_CAM = {"swinv2_tiny", "phikon_v2", "vit_large"}
_SWIN_RESHAPE = {"swinv2_tiny"}

# Classifier-head parameter names used for discriminative LR / freezing
HEAD_PATTERNS = {
    "resnet50": {"fc"},
    "convnext_tiny": {"head"},
    "swinv2_tiny": {"head"},
    "vit_large": {"head"},
    "phikon_v2": {"classifier"},
}


def create_model(name, num_classes, pretrained=True):
    """Instantiate a model by name and replace the classifier head."""
    name = _ALIASES.get(name, name)
    if name not in SUPPORTED_MODELS:
        raise ValueError(f"Unknown model '{name}'. Choose from {SUPPORTED_MODELS}")

    if name == "resnet50":
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        m = models.resnet50(weights=weights)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m

    if name == "convnext_tiny":
        return timm.create_model(
            "convnext_tiny.in12k_ft_in1k",
            pretrained=pretrained,
            num_classes=num_classes,
        )

    if name == "swinv2_tiny":
        return timm.create_model(
            "swinv2_cr_tiny_ns_224.sw_in1k",
            pretrained=pretrained,
            num_classes=num_classes,
        )

    if name == "vit_large":
        return timm.create_model(
            "vit_large_patch16_224.augreg_in21k_ft_in1k",
            pretrained=pretrained,
            num_classes=num_classes,
        )

    if name == "phikon_v2":
        return create_phikon_v2(num_classes, pretrained)

    raise ValueError(f"Unhandled model: {name}")


def get_model_info(name):
    """Return CAM target-layer path and reshape metadata for a model."""
    name = _ALIASES.get(name, name)
    if name not in SUPPORTED_MODELS:
        raise ValueError(f"Unknown model '{name}'")

    info = {
        "name": name,
        "requires_reshape_transform": name in _RESHAPE_CAM,
        "use_swin_reshape": name in _SWIN_RESHAPE,
        "input_size": 224,
    }

    cam_layers = {
        "resnet50":      ("layer4", None, None),
        "convnext_tiny": ("stages.3", None, None),
        "swinv2_tiny":   ("stages.3.blocks.1.norm1", 7, 7),
        "vit_large":     ("blocks.23.norm1", 14, 14),
        "phikon_v2":     ("backbone.encoder.layer.23.norm1", 14, 14),
    }
    layer, h, w = cam_layers[name]
    info["cam_target_layer"] = layer
    if h is not None:
        info["cam_reshape_height"] = h
        info["cam_reshape_width"] = w

    return info


def get_param_groups(model, model_name, base_lr, backbone_lr_factor=0.1):
    """Split parameters into backbone / head groups with different LRs."""
    model_name = _ALIASES.get(model_name, model_name)
    head_patterns = HEAD_PATTERNS.get(model_name, set())
    backbone, head = [], []

    for pname, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(pat in pname for pat in head_patterns):
            head.append(param)
        else:
            backbone.append(param)

    return [
        {"params": backbone, "lr": base_lr * backbone_lr_factor, "name": "backbone"},
        {"params": head, "lr": base_lr, "name": "head"},
    ]
