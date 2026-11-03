from dataclasses import dataclass, replace
import yaml
from pathlib import Path


@dataclass(frozen=True)
class ExperimentConfig:
    modality: str
    img_size: tuple[int, int, int]
    intensity_min: float
    intensity_max: float
    label_classes: int = 8
    image_channels: int = 1
    patch_size: tuple[int, int, int] = (16, 16, 16)
    spacing: tuple[float, float, float] = (1.5, 1.5, 2.0)
    nfolds: int = 5
    cv_seed: int = 12345
    train_seed: int = 0
    lr: float = 1e-4
    eval_interval: int = 500
    max_iterations: int = 27000
    warmup_epochs: int = 5
    train_batch_size: int = 1
    val_batch_size: int = 1
    sw_batch_size: int = 4
    include_background: bool = False
    cache_rate: float = 1.0

    @property
    def dataset_filename(self) -> str:
        return "dataset_1.json" if self.modality == "ct" else "dataset_0.json"

    def validate(self) -> None:
        if any(d <= 0 or d % p for d, p in zip(self.img_size, self.patch_size)):
            raise ValueError("Each image dimension must be a positive multiple of its patch size")
        if self.modality not in {"ct", "mri"}:
            raise ValueError("modality must be ct or mri")


def get_config(modality: str) -> ExperimentConfig:
    modality = modality.lower()
    if modality == "ct":
        return ExperimentConfig(
            modality="ct", img_size=(96, 96, 96),
            intensity_min=-150.0, intensity_max=1113.0,
            include_background=False,
        )
    if modality == "mri":
        return ExperimentConfig(
            modality="mri", img_size=(128, 128, 96),
            intensity_min=6.0, intensity_max=1200.0,
            include_background=True,
        )
    raise ValueError(f"Unknown modality {modality!r}; choose ct or mri")


def resolve_dataset_path(config: ExperimentConfig, dataset: str | None, data_dir: str) -> Path:
    return Path(dataset) if dataset else Path(data_dir) / config.dataset_filename


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    with path.open(encoding="utf-8") as file:
        values = yaml.safe_load(file)
    if not isinstance(values, dict) or "modality" not in values:
        raise ValueError("Config must contain a modality")
    config = get_config(values.pop("modality"))
    valid_fields = set(ExperimentConfig.__dataclass_fields__)
    unknown = set(values) - valid_fields
    if unknown:
        raise ValueError(f"Unknown config settings: {sorted(unknown)}")
    for key in ("img_size", "patch_size", "spacing"):
        if key in values:
            values[key] = tuple(values[key])
    config = replace(config, **values)
    config.validate()
    return config
