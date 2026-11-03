import numpy as np


def warmup_exponential_lr(init_lr: float, epoch: int, warmup_epochs: int, max_epochs: int) -> float:
    """Linear warm-up then the notebook's exponential decay."""
    if init_lr <= 0 or warmup_epochs <= 0 or max_epochs <= warmup_epochs:
        raise ValueError("Invalid learning rate / warmup / epoch schedule")
    if epoch < warmup_epochs:
        return init_lr * (epoch + 1) / warmup_epochs
    return init_lr * (0.1 ** ((epoch - warmup_epochs) / (max_epochs - warmup_epochs)))


def classwise_dice_numpy(y_true: np.ndarray, y_pred: np.ndarray, nclasses: int = 8):
    """Per-class Dice; an absent class in both masks is represented by nan."""
    if y_true.shape != y_pred.shape:
        raise ValueError("Shape mismatch")
    if nclasses < 1:
        raise ValueError("nclasses must be positive")
    result = []
    for c in range(nclasses):
        actual = y_true == c
        guessed = y_pred == c
        denom = actual.sum() + guessed.sum()
        result.append(float(2 * (actual & guessed).sum() / denom) if denom else float("nan"))
    return result
