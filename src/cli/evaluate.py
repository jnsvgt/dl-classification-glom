"""Test evaluation entry point."""

from src.cli import evaluate_parser
from src.evaluation.test_evaluation import TestEvaluator
from src.utils.logging import setup_logging, get_logger


def main():
    args = evaluate_parser().parse_args()
    setup_logging()
    logger = get_logger("evaluate")

    evaluator = TestEvaluator(
        checkpoint_path=args.checkpoint,
        test_data_dir=args.test_dir,
        model_name=args.model,
        num_classes=args.num_classes,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        augmentation=args.augmentation,
    )

    evaluator.setup()
    results = evaluator.evaluate()
    evaluator.save_results(results, args.output_dir)

    if results.metrics.report:
        print("\n" + results.metrics.report)


if __name__ == "__main__":
    main()
