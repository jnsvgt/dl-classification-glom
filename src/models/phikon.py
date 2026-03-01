"""Phikon-V2 wrapper — Owkin's histopathology ViT-L/16 (DINOv2 backbone)."""

import torch
import torch.nn as nn
from transformers import AutoModel


class PhikonClassifier(nn.Module):
    """ViT-L/16 backbone from Owkin with a linear classification head."""

    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        model_id = "owkin/phikon-v2"
        self.backbone = AutoModel.from_pretrained(model_id) if pretrained else AutoModel.from_config(
            AutoModel.from_pretrained(model_id).config
        )
        hidden_size = self.backbone.config.hidden_size  # 1024
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # DINOv2 returns last_hidden_state: (B, N+1, D). CLS token is index 0.
        features = self.backbone(x).last_hidden_state[:, 0]
        return self.classifier(features)


def create_phikon_v2(num_classes, pretrained=True):
    return PhikonClassifier(num_classes, pretrained)
