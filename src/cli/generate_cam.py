"""CAM generation entry point."""

import json
from pathlib import Path

import torch

from src.cli import cam_parser
from src.data.dataset import GlomerularDataset
from src.data.transforms import get_transforms
from src.evaluation.cam import generate_cam, visualize_cam, batch_generate_cam
from src.models.factory import create_model
from src.utils.logging import setup_logging, get_logger


def load_model_from_checkpoint(checkpoint_path, model_name, num_classes, device):
    model = create_model(model_name, num_classes)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    # strip torch.compile prefix
    cleaned = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(cleaned, strict=False)
    model.to(device)
    model.eval()
    return model


def main():
    args = cam_parser().parse_args()
    setup_logging()
    logger = get_logger("cam_cli")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = load_model_from_checkpoint(
        args.checkpoint, args.model, args.num_classes, device
    )

    val_transform = get_transforms(args.augmentation, is_training=False)
    dataset = GlomerularDataset(args.data_dir, transform=val_transform)
    class_names = dataset.classes

    logger.info(f"Model: {args.model} | Methods: {args.methods}")
    logger.info(f"Dataset: {len(dataset)} images, {len(class_names)} classes")

    # if a specific samples JSON is provided, only visualize those
    if args.samples_json:
        with open(args.samples_json) as f:
            samples = json.load(f)
        _generate_selected(model, dataset, samples, args, class_names, device, logger)
    else:
        batch_generate_cam(
            model=model,
            dataset=dataset,
            model_name=args.model,
            device=device,
            methods=args.methods,
            save_dir=args.output_dir,
            max_images=args.max_images,
            class_names=class_names,
        )


def _generate_selected(model, dataset, samples, args, class_names, device, logger):
    """Generate CAMs for hand-picked sample images."""
    output_dir = Path(args.output_dir)

    for entry in samples:
        idx = entry.get("index")
        if idx is None or idx >= len(dataset):
            continue

        img, label = dataset[idx]
        for method in args.methods:
            try:
                cam_map = generate_cam(model, img, label, method, args.model, device)
            except Exception as e:
                logger.warning(f"CAM failed: idx={idx}, method={method}: {e}")
                continue

            # predict
            with torch.no_grad():
                out = model(img.unsqueeze(0).to(device))
                pred = out.argmax(1).item()

            save_path = output_dir / method / f"{idx:04d}_{method}.png"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            visualize_cam(img, cam_map, label, pred, class_names,
                          save_path=save_path, method_name=method)

    logger.info(f"Selected CAMs saved to {output_dir}")


if __name__ == "__main__":
    main()
