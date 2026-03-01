"""Image transforms for training and evaluation.

Provides:
  - baseline (resize + normalise)
  - randaugment (histopathology-optimised, Faryna et al. 2021/2024)
  - manual (curated set following Tellez et al. 2019)

"""

import random

import numpy as np
from PIL import Image, ImageFilter, ImageOps, ImageEnhance
from torchvision import transforms

import torch

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ------------------------------------------------------------------
# Histopathology-optimised RandAugment (Faryna et al. 2021/2024)
# ------------------------------------------------------------------

class HistoRandAugment:
    """RandAugment adapted for histopathology images.

    Compared to vanilla RandAugment this version
    * drops posterize / solarize / invert (harm histo performance),
    * adds stain-shift, HSV-shift, Gaussian blur and noise,
    * samples magnitude continuously from U(0, m) per image.

    The stain-shift operation decomposes the image via the PAS stain
    matrix (Macenko decomposition) and randomly scales each stain
    channel, simulating batch-to-batch staining variation.
    """

    # Default PAS stain matrix (ImageJ "H PAS" preset, Ruifrok & Johnston 2001)
    # columns = [PAS-Magenta, Hematoxylin], rows = [R, G, B] optical density
    DEFAULT_STAIN_MATRIX = np.array([
        [0.1750, 0.6440],
        [0.9720, 0.7170],
        [0.1540, 0.2670],
    ], dtype=np.float32)

    DEFAULT_MAX_CONC = np.array([0.2602, 0.7619], dtype=np.float32)

    def __init__(self, num_ops=2, magnitude=5.0, stain_matrix=None):
        self.num_ops = num_ops
        self.magnitude = magnitude
        self.stain_matrix = np.asarray(stain_matrix, dtype=np.float32) if stain_matrix is not None else self.DEFAULT_STAIN_MATRIX
        self._stain_pinv = np.linalg.pinv(self.stain_matrix)

        self._ops = [
            ("identity",       self._identity,       False),
            ("autocontrast",   self._autocontrast,   False),
            ("equalize",       self._equalize,       False),
            ("rotate",         self._rotate,         True),
            ("shear_x",        self._shear_x,        True),
            ("shear_y",        self._shear_y,        True),
            ("translate_x",    self._translate_x,    True),
            ("translate_y",    self._translate_y,    True),
            ("brightness",     self._brightness,     True),
            ("color",          self._color,          True),
            ("contrast",       self._contrast,       True),
            ("sharpness",      self._sharpness,      True),
            ("stain_shift",    self._stain_shift,    True),
            ("hsv_shift",      self._hsv_shift,      True),
            ("gaussian_blur",  self._gaussian_blur,  True),
            ("gaussian_noise", self._gaussian_noise, True),
        ]

    # --- magnitude helpers ---------------------------------------------------

    def _mag(self):
        return random.uniform(0, self.magnitude)

    # --- augmentation operations ---------------------------------------------

    @staticmethod
    def _identity(img, _):      return img
    @staticmethod
    def _autocontrast(img, _):  return ImageOps.autocontrast(img)
    @staticmethod
    def _equalize(img, _):      return ImageOps.equalize(img)

    @staticmethod
    def _rotate(img, m):
        angle = random.uniform(-m * 6, m * 6)  # 0-15 → 0-90°
        return img.rotate(angle, resample=Image.BILINEAR, fillcolor=(128, 128, 128))

    @staticmethod
    def _shear_x(img, m):
        s = random.uniform(-m * 0.06, m * 0.06)
        return img.transform(img.size, Image.AFFINE, (1, s, 0, 0, 1, 0), fillcolor=(128, 128, 128))

    @staticmethod
    def _shear_y(img, m):
        s = random.uniform(-m * 0.06, m * 0.06)
        return img.transform(img.size, Image.AFFINE, (1, 0, 0, s, 1, 0), fillcolor=(128, 128, 128))

    @staticmethod
    def _translate_x(img, m):
        t = random.uniform(-m * 2, m * 2)
        return img.transform(img.size, Image.AFFINE, (1, 0, t, 0, 1, 0), fillcolor=(128, 128, 128))

    @staticmethod
    def _translate_y(img, m):
        t = random.uniform(-m * 2, m * 2)
        return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, t), fillcolor=(128, 128, 128))

    @staticmethod
    def _brightness(img, m):    return ImageEnhance.Brightness(img).enhance(max(0.05, m * 0.367))
    @staticmethod
    def _color(img, m):         return ImageEnhance.Color(img).enhance(max(0.05, m * 0.367))
    @staticmethod
    def _contrast(img, m):      return ImageEnhance.Contrast(img).enhance(max(0.05, m * 0.367))
    @staticmethod
    def _sharpness(img, m):     return ImageEnhance.Sharpness(img).enhance(max(0.05, m * 0.367))

    def _stain_shift(self, img, m):
        """Stain augmentation via Macenko decomposition in OD space."""
        arr = np.array(img, dtype=np.float32)
        od = -np.log10((arr + 1) / 256.0)
        h, w, _ = od.shape
        flat = od.reshape(-1, 3)

        conc = flat @ self._stain_pinv.T
        r = m * 0.06
        conc[:, 0] *= 1.0 + random.uniform(-r, r)
        conc[:, 1] *= 1.0 + random.uniform(-r, r)
        conc = np.maximum(conc, 0)

        od_aug = conc @ self.stain_matrix.T
        rgb = (256.0 * np.power(10, -od_aug) - 1).reshape(h, w, 3)
        return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))

    @staticmethod
    def _hsv_shift(img, m):
        arr = np.array(img.convert("HSV"), dtype=np.float32)
        r = m * 0.06
        for ch in range(3):
            arr[:, :, ch] = np.clip(arr[:, :, ch] + random.uniform(-r, r) * 255, 0, 255)
        return Image.fromarray(arr.astype(np.uint8), mode="HSV").convert("RGB")

    @staticmethod
    def _gaussian_blur(img, m):
        radius = m * 0.2
        return img.filter(ImageFilter.GaussianBlur(radius=radius)) if radius > 0.1 else img

    @staticmethod
    def _gaussian_noise(img, m):
        arr = np.array(img, dtype=np.float32)
        arr += np.random.normal(0, m * 1.5, arr.shape)
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    # --- main entry point ----------------------------------------------------

    def __call__(self, img):
        for _, fn, uses_m in random.choices(self._ops, k=self.num_ops):
            img = fn(img, self._mag() if uses_m else 0)
        return img

    def __repr__(self):
        return f"HistoRandAugment(num_ops={self.num_ops}, magnitude={self.magnitude})"


# ------------------------------------------------------------------
# Transform factories
# ------------------------------------------------------------------

def get_baseline_train_transforms(input_size=224):
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_baseline_val_transforms(input_size=224):
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_randaugment_transforms(input_size=224, num_ops=2, magnitude=5):
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        HistoRandAugment(num_ops=num_ops, magnitude=float(magnitude)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_manual_transforms(input_size=224, brightness=0.2, contrast=0.2,
                          saturation=0.0, hue=0.0, **_kw):
    """Curated augmentation set following Tellez et al. 2019 / Faryna et al. 2024."""
    return transforms.Compose([
        transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0), ratio=(0.8, 1.25)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=brightness, contrast=contrast,
                               saturation=saturation, hue=hue),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_transforms(augmentation_strategy, input_size=224, is_training=True,
                   config=None, **_kw):
    """Main factory: pick transforms by strategy name."""
    config = config or {}

    if not is_training:
        return get_baseline_val_transforms(input_size)

    if augmentation_strategy == "baseline":
        return get_baseline_train_transforms(input_size)
    elif augmentation_strategy == "randaugment":
        return get_randaugment_transforms(
            input_size, config.get("randaugment_n", 2), config.get("randaugment_m", 9),
        )
    elif augmentation_strategy == "manual":
        return get_manual_transforms(input_size, **config)
    else:
        raise ValueError(f"Unknown augmentation strategy: '{augmentation_strategy}'")


# ------------------------------------------------------------------
# Utility
# ------------------------------------------------------------------

def denormalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """Undo ImageNet normalisation for visualisation."""
    m = torch.tensor(mean).view(-1, 1, 1)
    s = torch.tensor(std).view(-1, 1, 1)
    if tensor.dim() == 4:
        m, s = m.unsqueeze(0), s.unsqueeze(0)
    return torch.clamp(tensor * s + m, 0, 1)
