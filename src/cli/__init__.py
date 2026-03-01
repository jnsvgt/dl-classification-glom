"""CLI argument parsers and helpers."""

import argparse


MODELS = ["resnet50", "convnext", "swinv2", "vit_large", "phikon_v2"]
OPTIMIZATIONS = ["baseline", "weighted_sampler", "weighted_ce", "ldam"]
AUGMENTATIONS = ["baseline", "randaugment", "manual"]


def base_parser(description=""):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--model", type=str, default="resnet50", choices=MODELS)
    p.add_argument("--num-classes", type=int, default=9)
    p.add_argument("--device", type=str, default="cuda")
    return p


def train_parser():
    p = base_parser("Train a glomerular classification model")

    # data
    p.add_argument("--data-dir", type=str, default="data_split_patient")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)

    # training
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--accumulation-steps", type=int, default=1)
    p.add_argument("--early-stopping", type=int, default=15,
                    help="Patience for early stopping")
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--gradient-clip", type=float, default=None)
    p.add_argument("--seed", type=int, default=42)

    # strategy
    p.add_argument("--optimization", type=str, default="baseline",
                    choices=OPTIMIZATIONS)
    p.add_argument("--augmentation", type=str, default="baseline",
                    choices=AUGMENTATIONS)

    # two-phase backbone freeze
    p.add_argument("--freeze-backbone-epochs", type=int, default=0,
                    help="Epochs to freeze backbone (0 to disable)")
    p.add_argument("--discriminative-lr", action="store_true",
                    help="Use lower LR for backbone layers")
    p.add_argument("--backbone-lr-factor", type=float, default=0.1)

    # LDAM
    p.add_argument("--ldam-drw-start", type=float, default=0.5,
                    help="Fraction of epochs before DRW activation")
    p.add_argument("--ldam-max-margin", type=float, default=0.5)

    # output
    p.add_argument("--checkpoint-dir", type=str, default="outputs/checkpoints")
    p.add_argument("--results-dir", type=str, default="outputs/results")
    p.add_argument("--config", type=str, default=None,
                    help="YAML config file (overrides CLI defaults)")

    return p


def evaluate_parser():
    p = base_parser("Evaluate model on test set")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--test-dir", type=str, required=True)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--output-dir", type=str, default="outputs/results")
    p.add_argument("--augmentation", type=str, default="baseline",
                    choices=AUGMENTATIONS)
    return p


def cam_parser():
    p = base_parser("Generate Class Activation Maps")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data-dir", type=str, required=True)
    p.add_argument("--output-dir", type=str, default="outputs/cam")
    p.add_argument("--methods", nargs="+", default=["gradcam"],
                    choices=["gradcam", "gradcam++", "eigencam", "layercam",
                             "scorecam", "hirescam", "attention_rollout"])
    p.add_argument("--max-images", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--augmentation", type=str, default="baseline",
                    choices=AUGMENTATIONS)
    p.add_argument("--samples-json", type=str, default=None,
                    help="JSON file specifying sample images to visualize")
    return p
