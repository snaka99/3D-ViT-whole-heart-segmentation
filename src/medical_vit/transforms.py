from monai.transforms import (
    Compose, CropForegroundd, DivisiblePadd, EnsureChannelFirstd,
    LoadImaged, Orientationd, RandAffined, RandGaussianNoised,
    RandRotated, Resized, ScaleIntensityRanged, Spacingd, ToTensord,
)
from .config import ExperimentConfig


def base_transforms(config: ExperimentConfig, with_label: bool = True) -> Compose:
    keys = ["image", "label"] if with_label else ["image"]
    modes = ("bilinear", "nearest") if with_label else "bilinear"
    resize_modes = ("trilinear", "nearest") if with_label else "trilinear"
    steps = [
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        CropForegroundd(keys=keys, source_key="image"),
        Orientationd(keys=keys, axcodes="RAS"),
        Spacingd(keys=keys, pixdim=config.spacing, mode=modes),
        ScaleIntensityRanged(
            keys=["image"], a_min=config.intensity_min,
            a_max=config.intensity_max, b_min=0.0, b_max=1.0, clip=True,
        ),
        DivisiblePadd(keys=keys, k=16),
        Resized(keys=keys, spatial_size=config.img_size, mode=resize_modes),
    ]
    steps.append(ToTensord(keys=keys))
    return Compose(steps)


def augmented_transforms(config: ExperimentConfig) -> Compose:
    """Original augmentations; MONAI transform defaults are preserved."""
    return Compose(
        list(base_transforms(config).transforms) + [
            RandAffined(keys=["image", "label"], prob=0.5, translate_range=10.0),
            RandRotated(keys=["image", "label"], prob=0.5, range_x=10.0),
            RandGaussianNoised(keys="image", prob=0.5),
        ]
    )
