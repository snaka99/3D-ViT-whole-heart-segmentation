from pathlib import Path

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt

def find_test_pairs(images_dir: Path, labels_dir: Path) -> list[dict]:
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise FileNotFoundError("Both image and label directories must exist")
    images = sorted(p for p in images_dir.iterdir() if p.name.endswith((".nii", ".nii.gz")))
    if not images:
        raise ValueError(f"No NIfTI images found in {images_dir}")
    pairs = []
    missing = []
    for image in images:
        label = labels_dir / image.name
        if not label.is_file():
            missing.append(image.name)
        else:
            pairs.append({"image": str(image), "label": str(label)})
    if missing:
        raise FileNotFoundError(
            "Labels must have the same filenames as the images. Missing: "
            + ", ".join(missing[:10])
        )
    return pairs


def surface_hd95(pred: np.ndarray, truth: np.ndarray) -> float:
    if not pred.any() or not truth.any():
        return float("nan")
    pred_surface = pred & ~binary_erosion(pred)
    true_surface = truth & ~binary_erosion(truth)
    dist_to_truth = distance_transform_edt(~true_surface)
    dist_to_pred = distance_transform_edt(~pred_surface)
    distances = np.concatenate([dist_to_truth[pred_surface], dist_to_pred[true_surface]])
    return float(np.percentile(distances, 95))


def case_metrics(pred: np.ndarray, truth: np.ndarray, num_classes: int,
                 include_background: bool = False) -> dict:
    if pred.shape != truth.shape:
        raise ValueError(f"Prediction and label shapes differ: {pred.shape} vs {truth.shape}")
    if not np.issubdtype(truth.dtype, np.integer):
        if not np.all(np.isfinite(truth)) or not np.all(truth == np.rint(truth)):
            raise ValueError("Ground-truth masks must contain integer class IDs")
        truth = truth.astype(np.int64)
    if truth.min() < 0 or truth.max() >= num_classes:
        raise ValueError(f"Ground-truth labels must be in [0, {num_classes - 1}]")
    if pred.min() < 0 or pred.max() >= num_classes:
        raise ValueError("Predictions contain out-of-range class IDs")
    result = {}
    dices, ious, distances = [], [], []
    for cls in range(0 if include_background else 1, num_classes):
        p, t = pred == cls, truth == cls
        npred, ntrue = int(p.sum()), int(t.sum())
        if not (npred or ntrue):
            dice = iou = hd95 = float("nan")
        else:
            common = int((p & t).sum())
            dice = 2 * common / (npred + ntrue)
            iou = common / (npred + ntrue - common)
            hd95 = surface_hd95(p, t)
        result[f"dice_class_{cls}"] = dice
        result[f"iou_class_{cls}"] = iou
        result[f"hd95_class_{cls}"] = hd95
        dices.append(dice)
        ious.append(iou)
        distances.append(hd95)
    result["mean_dice"] = float(np.nanmean(dices)) if np.isfinite(dices).any() else float("nan")
    result["mean_iou"] = float(np.nanmean(ious)) if np.isfinite(ious).any() else float("nan")
    result["mean_hd95_voxels"] = (
        float(np.nanmean(distances)) if np.isfinite(distances).any() else float("nan")
    )
    return result
