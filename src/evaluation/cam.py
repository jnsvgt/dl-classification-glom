"""Class Activation Map generation and visualization.

Supports GradCAM, GradCAM++, EigenCAM, LayerCAM, ScoreCAM,
HiResCAM, and Attention Rollout (for ViTs).
"""

import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from pytorch_grad_cam import (
    EigenCAM,
    GradCAM,
    GradCAMPlusPlus,
    HiResCAM,
    LayerCAM,
    ScoreCAM,
)
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import transforms

from src.data.transforms import denormalize
from src.models.factory import get_model_info
from src.utils.logging import get_logger

logger = get_logger("cam")

CAM_METHODS = {
    "gradcam": GradCAM,
    "gradcam++": GradCAMPlusPlus,
    "eigencam": EigenCAM,
    "layercam": LayerCAM,
    "scorecam": ScoreCAM,
    "hirescam": HiResCAM,
}


# -----------------------------------------------------------------
# Attention rollout (for ViTs that don't have conv feature maps)
# -----------------------------------------------------------------

class AttentionRollout:
    """Attention rollout for Vision Transformers (Abnar & Zuidema, 2020)."""

    def __init__(self, model, head_fusion="mean", discard_ratio=0.1):
        self.model = model
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio
        self._attentions = []
        self._hooks = []

    def _register_hooks(self):
        for module in self.model.modules():
            if hasattr(module, "attn_drop"):
                # timm ViT attention
                hook = module.register_forward_hook(self._save_attention)
                self._hooks.append(hook)

    @staticmethod
    def _save_attention(module, input, output):
        # timm stores attention weights in module.attn after softmax
        if hasattr(module, "attn"):
            module._saved_attn = module.attn.detach()

    def _collect_attentions(self):
        attentions = []
        for module in self.model.modules():
            if hasattr(module, "_saved_attn"):
                attentions.append(module._saved_attn)
                del module._saved_attn
        return attentions

    def _rollout(self, attentions):
        result = None
        for attn in attentions:
            # fuse heads
            if self.head_fusion == "mean":
                attn = attn.mean(dim=1)
            elif self.head_fusion == "max":
                attn = attn.max(dim=1).values
            elif self.head_fusion == "min":
                attn = attn.min(dim=1).values

            # discard low-attention tokens
            flat = attn.view(attn.size(0), -1)
            threshold = torch.quantile(flat, self.discard_ratio, dim=1, keepdim=True)
            attn = attn * (flat.view_as(attn) > threshold).float()

            # add identity + normalize
            I = torch.eye(attn.size(-1), device=attn.device).unsqueeze(0)
            attn = attn + I
            attn = attn / attn.sum(dim=-1, keepdim=True)

            result = attn if result is None else torch.bmm(attn, result)

        return result

    def __call__(self, input_tensor, target_class=None):
        self._register_hooks()
        try:
            self.model.eval()
            with torch.no_grad():
                output = self.model(input_tensor)
            attentions = self._collect_attentions()
        finally:
            for h in self._hooks:
                h.remove()
            self._hooks.clear()

        if not attentions:
            warnings.warn("No attention maps captured — model may not be a ViT")
            return np.zeros((input_tensor.shape[-2], input_tensor.shape[-1]))

        rollout = self._rollout(attentions)
        # use CLS token attention to patches
        mask = rollout[0, 0, 1:]
        n_patches = mask.shape[0]
        size = int(np.sqrt(n_patches))
        mask = mask.reshape(size, size).cpu().numpy()
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
        mask = cv2.resize(mask, (input_tensor.shape[-1], input_tensor.shape[-2]))
        return mask


# -----------------------------------------------------------------
# Main CAM generation
# -----------------------------------------------------------------

def generate_cam(model, image_tensor, target_class, method, model_name, device="cuda"):
    """Generate a CAM heatmap for a single image.

    Returns a 2D numpy array (H, W) in [0, 1].
    """
    model.eval()
    device = torch.device(device) if isinstance(device, str) else device
    image_tensor = image_tensor.unsqueeze(0).to(device) if image_tensor.dim() == 3 else image_tensor.to(device)

    if method == "attention_rollout":
        return AttentionRollout(model)(image_tensor, target_class)

    info = get_model_info(model_name)
    target_layers = info.get("cam_target_layers", [])
    if not target_layers:
        raise ValueError(f"No CAM target layers found for {model_name}")

    # resolve layer references
    resolved = []
    for layer_name in target_layers:
        obj = model
        for attr in layer_name.split("."):
            if attr.isdigit():
                obj = obj[int(attr)]
            else:
                obj = getattr(obj, attr)
        resolved.append(obj)

    cam_cls = CAM_METHODS.get(method)
    if cam_cls is None:
        raise ValueError(f"Unknown CAM method: {method}. Available: {list(CAM_METHODS)}")

    targets = [ClassifierOutputTarget(target_class)]
    with cam_cls(model=model, target_layers=resolved) as cam:
        grayscale = cam(input_tensor=image_tensor, targets=targets)

    return grayscale[0]


def visualize_cam(image_tensor, cam_map, true_label, pred_label, class_names,
                  save_path=None, method_name="GradCAM"):
    """Overlay CAM on image and optionally save."""
    img = denormalize(image_tensor)
    if isinstance(img, torch.Tensor):
        img = img.permute(1, 2, 0).cpu().numpy()
    img = np.clip(img, 0, 1).astype(np.float32)

    cam_map = cam_map.astype(np.float32)
    if cam_map.shape[:2] != img.shape[:2]:
        cam_map = cv2.resize(cam_map, (img.shape[1], img.shape[0]))

    overlay = show_cam_on_image(img, cam_map, use_rgb=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(img)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(cam_map, cmap="jet")
    axes[1].set_title(f"{method_name} heatmap")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    true_name = class_names[true_label] if true_label < len(class_names) else str(true_label)
    pred_name = class_names[pred_label] if pred_label < len(class_names) else str(pred_label)
    color = "green" if true_label == pred_label else "red"
    axes[2].set_title(f"True: {true_name}\nPred: {pred_name}", color=color)
    axes[2].axis("off")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


def batch_generate_cam(model, dataset, model_name, device="cuda",
                       methods=None, save_dir=None, max_images=50,
                       class_names=None):
    """Generate CAMs for a batch of images and save results."""
    if methods is None:
        methods = ["gradcam"]
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(device) if isinstance(device, str) else device
    model.eval()
    model.to(device)

    n = min(max_images, len(dataset))
    results = []

    for idx in range(n):
        img, label = dataset[idx]
        img_device = img.unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(img_device)
            pred = out.argmax(1).item()
            prob = torch.softmax(out, 1).max().item()

        for method in methods:
            try:
                cam_map = generate_cam(model, img, label, method, model_name, device)
            except Exception as e:
                logger.warning(f"CAM failed for image {idx}, method {method}: {e}")
                continue

            result = {
                "index": idx, "true_label": label, "pred_label": pred,
                "confidence": prob, "method": method, "cam_map": cam_map,
            }
            results.append(result)

            if save_dir and class_names:
                fname = f"{idx:04d}_{method}.png"
                sub = save_dir / method
                sub.mkdir(parents=True, exist_ok=True)
                visualize_cam(img, cam_map, label, pred, class_names,
                              save_path=sub / fname, method_name=method)

    logger.info(f"Generated {len(results)} CAM visualizations")
    return results
