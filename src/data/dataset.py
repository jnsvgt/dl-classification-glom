"""Dataset for glomerular morphology classification.

Loads images from a folder-per-class layout::

    data_dir/
    ├── Pattern1_Unauff\u00e4lliger Glomerulus/
    │   ├── img001.png
    │   └── ...
    ├── Pattern2_Amyloidose/
    └── ...
"""

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".pt"}


class GlomerularDataset(Dataset):

    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform

        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        # class names = sorted subfolder names
        self.class_names = sorted(
            d.name for d in self.data_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
            and any(f.suffix.lower() in _IMAGE_EXTS for f in d.iterdir())
        )
        if not self.class_names:
            raise ValueError(f"No class folders with images in {self.data_dir}")

        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}

        # (path, label) pairs
        self.samples = []
        for cname, cidx in self.class_to_idx.items():
            folder = self.data_dir / cname
            for f in folder.iterdir():
                if f.suffix.lower() in _IMAGE_EXTS:
                    self.samples.append((f, cidx))

        self._class_weights = None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        if path.suffix.lower() == ".pt":
            tensor = torch.load(path, weights_only=True)
            from torchvision.transforms.functional import to_pil_image
            img = to_pil_image(tensor)
        else:
            img = Image.open(path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label

    @property
    def num_classes(self):
        return len(self.class_names)

    # ------------------------------------------------------------------
    # Class-balance helpers
    # ------------------------------------------------------------------

    def get_class_counts(self):
        """Return {class_name: count} dict."""
        counts = {c: 0 for c in self.class_names}
        for _, cidx in self.samples:
            counts[self.class_names[cidx]] += 1
        return counts

    def get_class_counts_tensor(self):
        counts = self.get_class_counts()
        return torch.tensor([counts[c] for c in self.class_names], dtype=torch.float32)

    def get_class_weights(self):
        """Inverse-frequency weights, normalised to sum = num_classes."""
        if self._class_weights is None:
            counts = self.get_class_counts()
            total = sum(counts.values())
            w = torch.tensor(
                [total / counts[c] if counts[c] > 0 else 0.0 for c in self.class_names],
                dtype=torch.float32,
            )
            w = w / w.sum() * self.num_classes
            self._class_weights = w
        return self._class_weights

    def get_labels(self):
        return [label for _, label in self.samples]
