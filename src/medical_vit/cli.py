import argparse
import logging
from pathlib import Path

import torch

from .config import get_config, load_config, resolve_dataset_path


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Modular CT/MRI 3D ViT segmentation")
    p.add_argument("--modality", choices=["ct", "mri"], help="Use CT or MRI defaults")
    p.add_argument("--config", help="YAML experiment settings")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    commands = p.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Train the 5 cross-validation folds")
    train.add_argument("--data-dir", default="data")
    train.add_argument("--dataset", help="Explicit MONAI datalist .json file")
    train.add_argument("--output", help="Output dir, default results/<modality>")
    train.add_argument("--num-workers", type=int, default=0)
    train.add_argument("--max-iterations", type=int, default=None)
    train.add_argument("--eval-interval", type=int, default=None)
    train.add_argument("--checkpoint-interval", type=int, default=500)
    train.add_argument("--resume", action="store_true")
    predict = commands.add_parser("predict", help="Predict one NIfTI volume")
    predict.add_argument("--checkpoint", required=True)
    predict.add_argument("--image", required=True)
    predict.add_argument("--output", required=True)
    evaluate = commands.add_parser("evaluate", help="Evaluate labeled test NIfTI images")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--dataset", help="MONAI datalist JSON with labeled testing split")
    evaluate.add_argument("--images-dir", help="Folder of test images")
    evaluate.add_argument("--labels-dir", help="Folder of test labels (same filenames)")
    evaluate.add_argument("--output", default="results/test_evaluation")
    official = commands.add_parser("evaluate-official", help="MMWHS encrypted-label evaluation")
    official.add_argument("--predictions-dir", required=True)
    official.add_argument("--labels-dir", required=True)
    official.add_argument("--tools-dir", required=True)
    official.add_argument("--output", required=True)
    official.add_argument("--dry-run", action="store_true")
    plot = commands.add_parser("plot", help="Plot fold learning curves")
    plot.add_argument("--metrics", required=True)
    plot.add_argument("--output", required=True)
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if not args.modality and not args.config:
        parser().error("Provide --config or --modality")
    config = load_config(args.config) if args.config else get_config(args.modality)
    if args.config and args.modality and config.modality != args.modality:
        parser().error("--modality and YAML modality do not match")
    device_name = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no GPU with CUDA is available")
    device = torch.device(device_name)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.command == "train":
        from .training import run_training
        dataset_path = resolve_dataset_path(config, args.dataset, args.data_dir)
        if not dataset_path.is_file():
            raise FileNotFoundError(f"Dataset JSON not found: {dataset_path}")
        run_training(
            config, dataset_path, Path(args.output or f"results/{config.modality}"),
            device, num_workers=args.num_workers,
            max_iterations=args.max_iterations, eval_interval=args.eval_interval,
            checkpoint_interval=args.checkpoint_interval, resume=args.resume,
        )
    elif args.command == "predict":
        from .inference import predict_image
        output = predict_image(
            config, Path(args.checkpoint), Path(args.image), Path(args.output), device
        )
        print(f"Saved preprocessed-space segmentation: {output}")
    elif args.command == "evaluate":
        from .evaluation import evaluate_test_set
        result = evaluate_test_set(
            config, Path(args.checkpoint), Path(args.output), device,
            dataset=Path(args.dataset) if args.dataset else None,
            images_dir=Path(args.images_dir) if args.images_dir else None,
            labels_dir=Path(args.labels_dir) if args.labels_dir else None,
        )
        print(f"Saved evaluation results: {result}")
    elif args.command == "evaluate-official":
        from .mmwhs_official import evaluate_official
        result = evaluate_official(
            config.modality, Path(args.predictions_dir), Path(args.labels_dir),
            Path(args.tools_dir), Path(args.output), dry_run=args.dry_run,
        )
        print(f"Case manifest: {result}")
    elif args.command == "plot":
        from .visualization import plot_training_history
        output = plot_training_history(Path(args.metrics), Path(args.output))
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
