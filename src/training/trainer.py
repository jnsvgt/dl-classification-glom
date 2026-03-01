"""Training loop with AMP, gradient accumulation, and early stopping."""

import os
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.evaluation.metrics import MetricsSet, compute_all_metrics
from src.models.factory import HEAD_PATTERNS, get_param_groups
from src.utils.logging import get_logger

logger = get_logger("trainer")

CHECKPOINT_VERSION = 2


class CorruptedCheckpointError(Exception):
    pass


class MissingCheckpointKeyError(Exception):
    pass


_REQUIRED_KEYS = {
    "model_state_dict", "optimizer_state_dict", "scaler_state_dict",
    "epoch", "global_step", "best_metric", "best_epoch",
}


@dataclass
class TrainingState:
    epoch: int = 0
    global_step: int = 0
    best_metric: float = 0.0
    best_epoch: int = 0
    no_improve: int = 0


class Trainer:
    """End-to-end training loop.

    Supports mixed-precision (AMP), gradient accumulation, cosine LR
    schedule with linear warmup, early stopping, LDAM + DRW, backbone
    freezing / discriminative LR, and ``torch.compile``.
    """

    def __init__(self, config, model, criterion, device="cuda"):
        self.config = config
        self.device = torch.device(device) if isinstance(device, str) else device
        self.model = model.to(self.device)
        self.criterion = criterion.to(self.device)
        self.state = TrainingState()

        # channels-last for better GPU throughput
        if self.device.type == "cuda":
            self.model = self.model.to(memory_format=torch.channels_last)

        # torch.compile (PyTorch 2.0+)
        self._model_raw = self.model
        if self.device.type == "cuda" and hasattr(torch, "compile"):
            try:
                self.model = torch.compile(self.model, mode="default")
            except Exception:
                self._model_raw = self.model

        self.optimizer = self._build_optimizer()
        self.scheduler = None
        self.scaler = GradScaler("cuda")

        # LDAM / DRW bookkeeping
        from src.training.losses import LDAMLoss
        self.ldam_mode = isinstance(criterion, LDAMLoss)
        self._drw_active = False

        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Optimizer & scheduler
    # ------------------------------------------------------------------

    def _build_optimizer(self):
        lr = self.config.learning_rate
        if self.config.discriminative_lr_enabled:
            groups = get_param_groups(self.model, self.config.model_name,
                                     lr, self.config.backbone_lr_factor)
            for g in groups:
                g["weight_decay"] = self.config.weight_decay
            return AdamW(groups, fused=self.device.type == "cuda")

        # separate weight-decay from bias/norm params
        decay, no_decay = [], []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if "bias" in name or "norm" in name or "bn" in name:
                no_decay.append(p)
            else:
                decay.append(p)
        return AdamW([
            {"params": decay, "weight_decay": self.config.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ], lr=lr, fused=self.device.type == "cuda")

    def _build_scheduler(self, steps_per_epoch, epochs=None, warmup=None):
        total = (epochs or self.config.epochs) * steps_per_epoch
        warmup_steps = (warmup if warmup is not None else self.config.warmup_epochs) * steps_per_epoch
        warmup_sched = LinearLR(self.optimizer, start_factor=0.01, total_iters=max(1, warmup_steps))
        cosine = CosineAnnealingLR(self.optimizer, T_max=max(1, total - warmup_steps), eta_min=1e-6)
        return SequentialLR(self.optimizer, [warmup_sched, cosine], milestones=[warmup_steps])

    # ------------------------------------------------------------------
    # Backbone freeze / unfreeze (two-phase training)
    # ------------------------------------------------------------------

    def _freeze_backbone(self):
        patterns = HEAD_PATTERNS.get(self.config.model_name, set())
        for name, p in self.model.named_parameters():
            p.requires_grad = any(pat in name for pat in patterns)

    def _unfreeze_all(self):
        for p in self.model.parameters():
            p.requires_grad = True

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, train_loader, val_loader, class_names):
        """Run the full training loop. Returns MetricsSet of best epoch."""
        steps_per_epoch = max(1, len(train_loader) // self.config.accumulation_steps)

        # Phase-1 freeze?
        freeze_epochs = getattr(self.config, "freeze_backbone_epochs", 0)
        if freeze_epochs > 0:
            logger.info(f"Phase 1: backbone frozen for {freeze_epochs} epochs")
            self._freeze_backbone()
            self.optimizer = self._build_optimizer()

        self.scheduler = self._build_scheduler(
            steps_per_epoch,
            epochs=freeze_epochs if freeze_epochs > 0 else None,
        )

        best_metrics = None

        for epoch in range(self.config.epochs):
            self.state.epoch = epoch

            # Phase-2 transition
            if freeze_epochs > 0 and epoch == freeze_epochs:
                logger.info("Phase 2: unfreezing backbone")
                self._unfreeze_all()
                self.optimizer = self._build_optimizer()
                remaining = self.config.epochs - epoch
                phase2_warmup = getattr(self.config, "phase2_warmup_epochs", 2)
                self.scheduler = self._build_scheduler(steps_per_epoch, remaining, phase2_warmup)

            # DRW activation for LDAM
            if self.ldam_mode and not self._drw_active:
                drw_epoch = int(self.config.epochs * self.config.ldam_drw_start)
                if epoch >= drw_epoch:
                    from src.training.losses import compute_drw_weights
                    # fetch class counts from criterion buffer
                    # They were passed at construction time via cls_num_list
                    # Recompute from m_list is tricky; pass them via config instead
                    if hasattr(self.config, "_class_counts") and self.config._class_counts is not None:
                        w = compute_drw_weights(self.config._class_counts, device=self.device)
                        self.criterion.update_weight(w)
                        self._drw_active = True
                        logger.info(f"DRW activated at epoch {epoch}")

            t0 = time.time()
            train_loss = self._train_epoch(train_loader)
            val_loss, val_metrics = self._validate(val_loader, class_names)
            elapsed = time.time() - t0

            current = val_metrics.macro_f1
            if current > self.state.best_metric:
                self.state.best_metric = current
                self.state.best_epoch = epoch
                self.state.no_improve = 0
                val_metrics.best_epoch = epoch
                best_metrics = val_metrics
                self._save_checkpoint(self.checkpoint_dir / "best.pt", is_best=True, metrics=val_metrics)
                logger.info(f"  ★ New best Macro-F1 = {current:.4f}")
            else:
                self.state.no_improve += 1

            logger.info(
                f"Epoch {epoch:>3d}/{self.config.epochs} │ "
                f"train_loss {train_loss:.4f} │ val_loss {val_loss:.4f} │ "
                f"F1 {val_metrics.macro_f1:.4f} │ BalAcc {val_metrics.balanced_accuracy:.4f} │ "
                f"{elapsed:.0f}s"
            )

            if self.state.no_improve >= self.config.early_stopping_patience:
                logger.info(f"Early stopping after {epoch + 1} epochs")
                break

        # cleanup
        self._save_checkpoint(self.checkpoint_dir / "last.pt", is_best=False,
                              metrics=val_metrics if val_metrics else best_metrics)
        for f in self.checkpoint_dir.glob("*.pt"):
            if f.name != "best.pt":
                f.unlink(missing_ok=True)

        if best_metrics is None:
            raise RuntimeError("Training finished without producing valid metrics")
        return best_metrics

    # ------------------------------------------------------------------
    # one epoch
    # ------------------------------------------------------------------

    def _train_epoch(self, loader):
        self.model.train()
        total_loss, n_batches = 0.0, 0
        self.optimizer.zero_grad()

        pbar = tqdm(loader, desc=f"Epoch {self.state.epoch}", leave=False)
        for i, (imgs, targets) in enumerate(pbar):
            imgs = imgs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            if self.device.type == "cuda":
                imgs = imgs.to(memory_format=torch.channels_last)

            with autocast("cuda", dtype=torch.float16):
                loss = self.criterion(self.model(imgs), targets)
                loss = loss / self.config.accumulation_steps

            if torch.isnan(loss):
                raise RuntimeError(f"NaN loss at epoch {self.state.epoch}, batch {i}")

            self.scaler.scale(loss).backward()

            if (i + 1) % self.config.accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                clip = getattr(self.config, "gradient_clip_norm", None)
                if clip:
                    nn.utils.clip_grad_norm_(self.model.parameters(), clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.state.global_step += 1

                if self.scheduler is not None:
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=UserWarning,
                                                module="torch.optim.lr_scheduler")
                        self.scheduler.step()

            total_loss += loss.item() * self.config.accumulation_steps
            n_batches += 1
            pbar.set_postfix(loss=f"{total_loss / n_batches:.4f}",
                             lr=f"{self.optimizer.param_groups[0]['lr']:.2e}")

        return total_loss / max(n_batches, 1)

    # ------------------------------------------------------------------
    # validation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _validate(self, loader, class_names):
        self.model.eval()
        total_loss = 0.0
        all_preds, all_targets, all_probs = [], [], []

        for imgs, targets in tqdm(loader, desc="Validating", leave=False):
            imgs = imgs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            if self.device.type == "cuda":
                imgs = imgs.to(memory_format=torch.channels_last)

            with autocast("cuda", dtype=torch.float16):
                out = self.model(imgs)
                total_loss += self.criterion(out, targets).item()

            probs = torch.softmax(out.float(), dim=1)
            all_preds.append(probs.argmax(1).cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_targets)
        y_proba = np.concatenate(all_probs)
        val_loss = total_loss / len(loader)

        metrics = compute_all_metrics(y_true, y_pred, y_proba, class_names,
                                      val_loss=val_loss, best_epoch=self.state.best_epoch)
        return val_loss, metrics

    # ------------------------------------------------------------------
    # checkpoints
    # ------------------------------------------------------------------

    def _save_checkpoint(self, path, is_best=False, metrics=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        ckpt = {
            "epoch": self.state.epoch,
            "global_step": self.state.global_step,
            "model_state_dict": self._model_raw.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "scaler_state_dict": self.scaler.state_dict(),
            "best_metric": self.state.best_metric,
            "best_epoch": self.state.best_epoch,
            "epochs_without_improvement": self.state.no_improve,
            "config": self.config.to_dict(),
            "checkpoint_version": CHECKPOINT_VERSION,
            "timestamp": datetime.now().isoformat(),
            "is_best": is_best,
        }
        if metrics:
            ckpt["metrics"] = metrics.to_dict()
        tmp = path.with_suffix(".pt.tmp")
        torch.save(ckpt, tmp)
        os.replace(tmp, path)

    def resume_from_checkpoint(self, path):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        try:
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
        except Exception as e:
            raise CorruptedCheckpointError(f"Cannot load {path}: {e}")

        missing = _REQUIRED_KEYS - set(ckpt.keys())
        if missing:
            raise MissingCheckpointKeyError(f"Missing keys: {missing}")

        self._model_raw.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if ckpt.get("scheduler_state_dict") and self.scheduler:
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.scaler.load_state_dict(ckpt["scaler_state_dict"])

        self.state.epoch = ckpt["epoch"] + 1
        self.state.global_step = ckpt["global_step"]
        self.state.best_metric = ckpt["best_metric"]
        self.state.best_epoch = ckpt["best_epoch"]
        self.state.no_improve = ckpt.get("epochs_without_improvement", 0)

        logger.info(f"Resumed from epoch {ckpt['epoch']}, best_metric={self.state.best_metric:.4f}")

    def cleanup(self):
        """Release GPU memory after training finishes."""
        if hasattr(self, "_model_raw"):
            self._model_raw.cpu()
        if hasattr(self, "model"):
            try:
                self.model.cpu()
            except Exception:
                pass
        if torch.cuda.is_available():
            try:
                torch._dynamo.reset()
            except Exception:
                pass
