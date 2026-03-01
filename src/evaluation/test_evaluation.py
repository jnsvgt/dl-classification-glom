"""Test-set evaluation: load a checkpoint, run inference, save results."""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import GlomerularDataset
from src.data.transforms import get_transforms
from src.evaluation.metrics import MetricsSet, compute_all_metrics, plot_confusion_matrix
from src.models.factory import create_model
from src.utils.logging import get_logger

logger = get_logger("test_eval")


@dataclass
class TestResults:
    metrics: MetricsSet = field(default_factory=MetricsSet)
    predictions: np.ndarray = field(default_factory=lambda: np.array([]))
    targets: np.ndarray = field(default_factory=lambda: np.array([]))
    probabilities: np.ndarray = field(default_factory=lambda: np.array([]))
    file_paths: List[str] = field(default_factory=list)
    inference_time: float = 0.0


class TestEvaluator:
    """Load a trained model checkpoint and evaluate on a test set."""

    def __init__(self, checkpoint_path, test_data_dir, model_name,
                 num_classes=9, batch_size=32, num_workers=4, device="cuda",
                 augmentation="baseline"):
        self.checkpoint_path = Path(checkpoint_path)
        self.test_data_dir = Path(test_data_dir)
        self.model_name = model_name
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = torch.device(device)
        self.augmentation = augmentation

        self.model = None
        self.dataset = None

    def setup(self):
        """Load model from checkpoint and prepare test dataset."""
        # build model
        self.model = create_model(self.model_name, self.num_classes)
        self._load_checkpoint()
        self.model.to(self.device)
        self.model.eval()

        # build dataset
        val_transform = get_transforms(self.augmentation, is_training=False)
        self.dataset = GlomerularDataset(self.test_data_dir, transform=val_transform)
        logger.info(f"Test set: {len(self.dataset)} images, {len(self.dataset.classes)} classes")

    def _load_checkpoint(self):
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        ckpt = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)

        # handle compiled model keys (_orig_mod prefix)
        cleaned = {}
        for k, v in state.items():
            k_clean = k.replace("_orig_mod.", "")
            cleaned[k_clean] = v

        missing, unexpected = self.model.load_state_dict(cleaned, strict=False)
        if missing:
            logger.warning(f"Missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
        if unexpected:
            logger.warning(f"Unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

        # log training info
        if "epoch" in ckpt:
            logger.info(f"Loaded checkpoint from epoch {ckpt['epoch']}")
        if "best_metric" in ckpt:
            logger.info(f"Best metric (val): {ckpt['best_metric']:.4f}")

    @torch.no_grad()
    def evaluate(self) -> TestResults:
        """Run inference on test set and compute metrics."""
        if self.model is None:
            self.setup()

        loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        all_preds, all_targets, all_probs = [], [], []
        t0 = time.time()

        for imgs, targets in tqdm(loader, desc="Testing"):
            imgs = imgs.to(self.device, non_blocking=True)
            out = self.model(imgs)
            probs = torch.softmax(out.float(), dim=1)
            all_preds.append(probs.argmax(1).cpu().numpy())
            all_targets.append(targets.numpy())
            all_probs.append(probs.cpu().numpy())

        elapsed = time.time() - t0
        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_targets)
        y_proba = np.concatenate(all_probs)

        class_names = self.dataset.classes
        metrics = compute_all_metrics(y_true, y_pred, y_proba, class_names)

        results = TestResults(
            metrics=metrics,
            predictions=y_pred,
            targets=y_true,
            probabilities=y_proba,
            file_paths=[str(p) for p, _ in self.dataset.samples],
            inference_time=elapsed,
        )

        logger.info(f"Test results: F1={metrics.macro_f1:.4f}  "
                     f"BalAcc={metrics.balanced_accuracy:.4f}  "
                     f"({elapsed:.1f}s)")
        return results

    def save_results(self, results: TestResults, output_dir):
        """Save metrics, confusion matrix, and per-image predictions."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # metrics json
        with open(out / "test_metrics.json", "w") as f:
            json.dump(results.metrics.to_dict(), f, indent=2)

        # classification report
        with open(out / "classification_report.txt", "w") as f:
            f.write(results.metrics.report)

        # confusion matrix
        if results.metrics.confusion_mat is not None:
            plot_confusion_matrix(
                results.metrics.confusion_mat,
                self.dataset.classes,
                save_path=out / "confusion_matrix.png",
            )

        # per-image predictions
        preds_list = []
        for i in range(len(results.predictions)):
            entry = {
                "file": results.file_paths[i] if i < len(results.file_paths) else "",
                "true": int(results.targets[i]),
                "pred": int(results.predictions[i]),
                "correct": bool(results.targets[i] == results.predictions[i]),
            }
            if results.probabilities is not None and len(results.probabilities) > i:
                entry["confidence"] = float(results.probabilities[i].max())
            preds_list.append(entry)

        with open(out / "predictions.json", "w") as f:
            json.dump(preds_list, f, indent=2)

        # misclassified
        misclassified = [p for p in preds_list if not p["correct"]]
        with open(out / "misclassified.json", "w") as f:
            json.dump(misclassified, f, indent=2)

        logger.info(f"Results saved to {out}")
        logger.info(f"Misclassified: {len(misclassified)}/{len(preds_list)} "
                     f"({100 * len(misclassified) / max(len(preds_list), 1):.1f}%)")
