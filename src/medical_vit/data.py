from abc import ABC, abstractmethod
import sys

from monai.apps import CrossValidation
from monai.data import CacheDataset, DataLoader, load_decathlon_datalist
from torch.utils.data import ConcatDataset

from .config import ExperimentConfig
from .transforms import augmented_transforms, base_transforms


class CVDataset(ABC, CacheDataset):
    """Adapted from the CVDataset in both supplied notebooks."""

    def __init__(self, data, transform, cache_num=sys.maxsize, cache_rate=1.0, num_workers=4):
        data = self._split_datalist(datalist=data)
        super().__init__(data, transform, cache_num=cache_num, cache_rate=cache_rate,
                         num_workers=num_workers)

    @abstractmethod
    def _split_datalist(self, datalist):
        raise NotImplementedError


def read_split(json_path, split: str):
    try:
        return load_decathlon_datalist(str(json_path), True, split)
    except KeyError as error:
        raise ValueError(f"Dataset JSON lacks split {split!r}: {json_path}") from error


def _paths(items):
    return {str(item["image"]) for item in items}


def make_cv_loaders(config: ExperimentConfig, json_path, num_workers: int = 0):
    """Yield (fold_index_1_based, training_loader, validation_loader).

    Training set: normal + augmented version of ONLY training records.
    MONAI CrossValidation remains responsible for fold assignment as originally.
    """
    # Augment only training cases so validation images never enter training.
    records = read_split(json_path, "training")
    if len(records) < config.nfolds:
        raise ValueError("Not enough training records for cross-validation folds")
    cvdataset = CrossValidation(
        dataset_cls=CVDataset, data=records, nfolds=config.nfolds,
        seed=config.cv_seed, transform=base_transforms(config),
    )
    folds = list(range(config.nfolds))
    for i in folds:
        training = cvdataset.get_dataset(folds=folds[:i] + folds[i+1:])
        validation = cvdataset.get_dataset(folds=i, transform=base_transforms(config))
        training_records = training.data
        validation_records = validation.data
        overlap = _paths(training_records) & _paths(validation_records)
        if overlap:
            raise ValueError(f"Cross-validation leakage detected: {len(overlap)} overlapping images")
        augmentation = CacheDataset(
            data=training_records, transform=augmented_transforms(config),
            cache_rate=config.cache_rate, num_workers=num_workers,
        )
        combined = ConcatDataset([training, augmentation])
        train_loader = DataLoader(
            combined, batch_size=config.train_batch_size, shuffle=True,
            num_workers=num_workers,
        )
        val_loader = DataLoader(
            validation, batch_size=config.val_batch_size, num_workers=num_workers,
        )
        yield i + 1, train_loader, val_loader


def make_split_loader(config, json_path, split, with_label=True, num_workers=0):
    dataset = CacheDataset(
        data=read_split(json_path, split),
        transform=base_transforms(config, with_label=with_label),
        cache_rate=config.cache_rate, num_workers=num_workers,
    )
    return DataLoader(dataset, batch_size=1, num_workers=num_workers)
