import numpy as np
import pytest

from medical_vit.evaluation_metrics import case_metrics, find_test_pairs


def test_perfect_prediction():
    mask = np.zeros((8, 8, 8), dtype=np.int32)
    mask[2:5, 2:5, 2:5] = 1
    values = case_metrics(mask, mask, num_classes=2)
    assert values["mean_dice"] == 1.0
    assert values["mean_iou"] == 1.0
    assert values["mean_hd95_voxels"] == 0.0


def test_wrong_prediction():
    truth = np.zeros((8, 8, 8), dtype=np.int32)
    pred = np.zeros_like(truth)
    truth[:2] = 1
    pred[6:] = 1
    values = case_metrics(pred, truth, num_classes=2)
    assert values["mean_dice"] == 0.0
    assert values["mean_iou"] == 0.0
    assert values["mean_hd95_voxels"] > 0.0


def test_invalid_labels():
    with pytest.raises(ValueError):
        case_metrics(np.zeros((2, 2, 2), dtype=int),
                     np.full((2, 2, 2), 8, dtype=int), 8)


def test_matching_names(tmp_path):
    images, labels = tmp_path / "images", tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    (images / "case_001.nii.gz").touch()
    with pytest.raises(FileNotFoundError):
        find_test_pairs(images, labels)
    (labels / "case_001.nii.gz").touch()
    pairs = find_test_pairs(images, labels)
    assert len(pairs) == 1
    assert pairs[0]["image"].endswith("case_001.nii.gz")
