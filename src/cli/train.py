"""Training entry point."""

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

from src.cli import train_parser
from src.data.dataset import GlomerularDataset
from src.data.transforms import get_transforms
from src.models.factory import create_model
from src.training.config import TrainingConfig
from src.training.losses import create_loss, wrap_model_with_ldam
from src.training.samplers import create_weighted_sampler
from src.training.trainer import Trainer
from src.utils.logging import get_logger, setup_logging


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_config(args):
    """Build TrainingConfig from CLI args + optional YAML file."""
    # start from YAML if provided
    if args.config:
        with open(args.config) as f:
            yaml_cfg = yaml.safe_load(f) or {}
    else:
        yaml_cfg = {}

    # CLI overrides YAML
    cfg = TrainingConfig(
        model_name=args.model,
        num_classes=args.num_classes,
        optimization=args.optimization,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        batch_size=args.batch_size,
        accumulation_steps=args.accumulation_steps,
        warmup_epochs=args.warmup_epochs,
        early_stopping_patience=args.early_stopping,
        label_smoothing=args.label_smoothing,
        gradient_clip_norm=args.gradient_clip,
        augmentation=args.augmentation,
        freeze_backbone_epochs=args.freeze_backbone_epochs,
        discriminative_lr_enabled=args.discriminative_lr,
        backbone_lr_factor=args.backbone_lr_factor,
        ldam_drw_start=args.ldam_drw_start,
        ldam_max_margin=args.ldam_max_margin,
        data_dir=args.data_dir,
        num_workers=args.num_workers,
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
        results_dir=args.results_dir,
    )

    # apply YAML overrides for fields not explicitly set via CLI
    for k, v in yaml_cfg.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)

    return cfg


def main():
    args = train_parser().parse_args()
    setup_logging()
    logger = get_logger("train")

    config = build_config(args)
    set_seed(config.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    logger.info(f"Model: {config.model_name} | Opt: {config.optimization} | Aug: {config.augmentation}")

    # ---- data
    train_transform = get_transforms(config.augmentation, is_training=True)
    val_transform = get_transforms(config.augmentation, is_training=False)

    train_dir = Path(config.data_dir) / "train"
    val_dir = Path(config.data_dir) / "val"
    train_ds = GlomerularDataset(train_dir, transform=train_transform)
    val_ds = GlomerularDataset(val_dir, transform=val_transform)
    class_names = train_ds.classes

    logger.info(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Classes: {len(class_names)}")
    config._class_names = class_names
    config._class_counts = [train_ds.class_counts.get(c, 0) for c in class_names]

    # ---- sampler
    sampler = None
    shuffle = True
    if config.optimization == "weighted_sampler":
        sampler = create_weighted_sampler(train_ds)
        shuffle = False

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=shuffle,
        num_workers=config.num_workers, pin_memory=True, sampler=sampler,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=True,
    )

    # ---- model
    model = create_model(config.model_name, config.num_classes)

    # LDAM: replace final layer with NormedLinear
    if config.optimization == "ldam":
        model = wrap_model_with_ldam(model, config.model_name, config.num_classes)

    # ---- loss
    criterion = create_loss(
        config.optimization,
        class_counts=config._class_counts,
        num_classes=config.num_classes,
        label_smoothing=config.label_smoothing,
        max_margin=config.ldam_max_margin,
    )

    # ---- train
    trainer = Trainer(config, model, criterion, device=device)

    logger.info(f"Starting training for {config.epochs} epochs")
    logger.info(f"Config: {json.dumps(config.to_dict(), indent=2)}")

    try:
        best_metrics = trainer.fit(train_loader, val_loader, class_names)
    except KeyboardInterrupt:
        logger.info("Training interrupted")
        sys.exit(1)
    finally:
        trainer.cleanup()

    # ---- save results
    results_dir = Path(config.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "config": config.to_dict(),
        "metrics": best_metrics.to_dict(),
    }
    with open(results_dir / "training_results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Best Macro-F1: {best_metrics.macro_f1:.4f} (epoch {best_metrics.best_epoch})")
    logger.info(f"Checkpoint: {config.checkpoint_dir}/best.pt")
    logger.info(f"Results: {results_dir}/training_results.json")

    # print report
    if best_metrics.report:
        print("\n" + best_metrics.report)


if __name__ == "__main__":
    main()
