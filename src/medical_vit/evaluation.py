import csv
import json
from pathlib import Path

import numpy as np
import torch
from monai.inferers import sliding_window_inference

from .evaluation_metrics import find_test_pairs, case_metrics
from .config import ExperimentConfig
from .data import read_split
from .inference import load_model
from .transforms import base_transforms


def evaluate_test_set(config: ExperimentConfig, checkpoint: Path, output_dir: Path,
                      device: torch.device, dataset: Path | None = None,
                      images_dir: Path | None = None, labels_dir: Path | None = None) -> Path:
    if dataset is not None:
        pairs = read_split(dataset, "testing")
        if not pairs or any("label" not in p for p in pairs):
            raise ValueError("The testing split must contain image and label paths")
    elif images_dir is not None and labels_dir is not None:
        pairs = find_test_pairs(images_dir, labels_dir)
    else:
        raise ValueError("Provide --dataset, or both --images-dir and --labels-dir")

    model = load_model(config, Path(checkpoint), device)
    transforms = base_transforms(config, with_label=True)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in pairs:
        data = transforms(case)
        image = data["image"].unsqueeze(0).to(device)
        truth = data["label"].squeeze(0).cpu().numpy()
        with torch.no_grad():
            logits = sliding_window_inference(
                image, config.img_size, config.sw_batch_size, model
            )
            prediction = logits.argmax(dim=1)[0].cpu().numpy()
        row = {"case": Path(case["image"]).name}
        row.update(case_metrics(prediction, truth, config.label_classes,
                                config.include_background))
        rows.append(row)
        print(f"{row['case']}: Dice={row['mean_dice']:.4f} IoU={row['mean_iou']:.4f}")

    csv_path = output_dir / "test_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "number_of_cases": len(rows),
        "checkpoint": str(checkpoint),
        "modality": config.modality,
        "space": "resized preprocessed voxel grid",
        "hd95_unit": "processed voxels (not mm)",
    }
    for name in ("mean_dice", "mean_iou", "mean_hd95_voxels"):
        values = np.array([row[name] for row in rows], dtype=float)
        valid = values[np.isfinite(values)]
        summary[name] = float(valid.mean()) if len(valid) else None
        summary[f"std_{name}"] = float(valid.std(ddof=1)) if len(valid) > 1 else None
    (output_dir / "test_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return csv_path
