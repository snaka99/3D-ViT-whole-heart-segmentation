import numpy as np
import pytest


def test_scheduler_respects_original_formula():
    from medical_vit.math_utils import warmup_exponential_lr
    assert warmup_exponential_lr(1e-4, 0, 5, 100) == pytest.approx(2e-5)
    assert warmup_exponential_lr(1e-4, 4, 5, 100) == pytest.approx(1e-4)
    assert warmup_exponential_lr(1e-4, 100, 5, 100) == pytest.approx(1e-5)


def test_dice_for_perfect_masks_and_empty_class():
    from medical_vit.math_utils import classwise_dice_numpy
    mask = np.array([[0, 1], [1, 0]])
    scores = classwise_dice_numpy(mask, mask, nclasses=3)
    assert scores[:2] == [1.0, 1.0]
    assert np.isnan(scores[2])


def test_dice_rejects_wrong_shape():
    from medical_vit.math_utils import classwise_dice_numpy
    with pytest.raises(ValueError):
        classwise_dice_numpy(np.zeros((2, 2)), np.zeros((3, 3)))


def test_smoothing_matches_notebook():
    from medical_vit.visualization import smooth_loss
    assert smooth_loss([1.0, 0.0, 0.0], 0.9) == pytest.approx([1.0, 0.9, 0.81])
