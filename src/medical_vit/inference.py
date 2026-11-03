from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from nibabel.processing import resample_from_to
from monai.inferers import sliding_window_inference

from .config import ExperimentConfig
from .model import ViTSegmentation
from .transforms import base_transforms


def load_model(config: ExperimentConfig, checkpoint: Path, device: torch.device):
    model = ViTSegmentation(
        in_channels=config.image_channels, img_size=config.img_size,
        patch_size=config.patch_size, num_classes=config.label_classes,
    ).to(device)
    try:
        state_dict = torch.load(str(checkpoint), map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(str(checkpoint), map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict_image(
    config: ExperimentConfig,
    checkpoint: Path,
    image_path: Path,
    output: Path,
    device: torch.device,
    sw_overlap: float = 0.25,
) -> Path:
    """Predict segmentation and resample the mask to the input NIfTI grid."""
    model = load_model(config, checkpoint, device)
    sample = base_transforms(config, with_label=False)({"image": str(image_path)})
    image = sample["image"]
    if not hasattr(image, "affine"):
        raise ValueError("The preprocessed image has no spatial affine metadata.")

    inputs = image.unsqueeze(0).to(device)
    with torch.no_grad():
        logits = sliding_window_inference(
            inputs, config.img_size, config.sw_batch_size, model, overlap=sw_overlap
        )
        prediction = torch.argmax(logits, dim=1).cpu().numpy()[0].astype(np.uint8)

    preprocessed_affine = image.affine.detach().cpu().numpy()
    original = nib.load(str(image_path))
    mask_in_processed_space = nib.Nifti1Image(prediction, preprocessed_affine)

    # Nearest-neighbor resampling preserves discrete segmentation labels.
    restored = resample_from_to(
        mask_in_processed_space,
        (original.shape[:3], original.affine),
        order=0,
        mode="constant",
        cval=0,
    )
    restored_data = np.asarray(restored.dataobj).astype(np.uint8)

    output.parent.mkdir(parents=True, exist_ok=True)
    header = original.header.copy()
    header.set_data_dtype(np.uint8)
    nib.save(nib.Nifti1Image(restored_data, original.affine, header=header), str(output))
    return output
