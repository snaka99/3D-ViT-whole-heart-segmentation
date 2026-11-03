import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def smooth_loss(loss_values, smoothing_factor=0.9):
    if not 0 <= smoothing_factor <= 1:
        raise ValueError("smoothing_factor must be in [0,1]")
    if not loss_values:
        return []
    smoothed = [float(loss_values[0])]
    for val in loss_values[1:]:
        smoothed.append(smoothing_factor * smoothed[-1] + (1 - smoothing_factor) * val)
    return smoothed


def plot_training_history(metrics_json: Path, output: Path) -> Path:
    history = json.loads(metrics_json.read_text(encoding="utf-8"))
    steps = history["evaluation_steps"]
    losses = smooth_loss(history["loss_values"])
    dice = history["mean_dice_values"]
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(steps, losses)
    axes[0].set(title="Smoothed training loss", xlabel="Iteration", ylabel="Loss")
    axes[1].plot(steps, dice)
    axes[1].set(title="Validation mean Dice", xlabel="Iteration", ylabel="Dice")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return output


def plot_segmentation(image, prediction, output, label=None, slice_index=None):
    """Image and mask arrays must be preprocessed 3D volumes on the same grid."""
    image = np.asarray(image).squeeze()
    prediction = np.asarray(prediction).squeeze()
    label = None if label is None else np.asarray(label).squeeze()
    if image.shape != prediction.shape or (label is not None and label.shape != image.shape):
        raise ValueError("Shape mismatch among image, prediction, and optional label")
    idx = image.shape[-1] // 2 if slice_index is None else slice_index
    if not 0 <= idx < image.shape[-1]:
        raise IndexError("Requested slice is outside the volume")
    imgs = [("Image", image), ("Prediction", prediction)]
    if label is not None:
        imgs.insert(1, ("Ground truth", label))
    fig, axes = plt.subplots(1, len(imgs), figsize=(6 * len(imgs), 5))
    for ax, (name, volume) in zip(axes, imgs):
        ax.imshow(volume[:, :, idx], cmap="gray" if name == "Image" else None)
        ax.set_title(name)
        ax.axis("off")
    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output
