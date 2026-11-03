import numpy as np
import torch
from monai.data import decollate_batch
from monai.metrics import (
    DiceMetric, MeanIoU, HausdorffDistanceMetric, ConfusionMatrixMetric,
)
from monai.transforms import AsDiscrete

from .config import ExperimentConfig


class MetricCollector:
    def __init__(self, config: ExperimentConfig):
        self.num_classes = config.label_classes
        self.pred_post = AsDiscrete(argmax=True, to_onehot=self.num_classes)
        self.label_post = AsDiscrete(to_onehot=self.num_classes)
        self.dice = DiceMetric(include_background=config.include_background, reduction="mean")
        self.iou = MeanIoU(include_background=config.include_background, reduction="mean",
                           ignore_empty=True)
        self.hausdorff = HausdorffDistanceMetric(
            include_background=False, distance_metric="euclidean", percentile=None,
            directed=False, reduction="mean",
        )
        self.confusion = ConfusionMatrixMetric(
            include_background=config.include_background,
            metric_name=("accuracy", "sensitivity", "precision", "recall"),
            compute_sample=True, reduction="mean",
        )

    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        preds = [self.pred_post(x) for x in decollate_batch(logits)]
        truth = [self.label_post(x) for x in decollate_batch(labels)]
        self.dice(y_pred=preds, y=truth)
        self.iou(y_pred=preds, y=truth)
        self.hausdorff(y_pred=preds, y=truth)
        self.confusion(y_pred=preds, y=truth)

    @staticmethod
    def _to_jsonable(value):
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return float(value.detach().cpu().item())
            return value.detach().cpu().numpy().tolist()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (tuple, list)):
            return [MetricCollector._to_jsonable(v) for v in value]
        if isinstance(value, (float, int, str)):
            return value
        raise TypeError(f"Unsupported metric output: {type(value)}")

    def aggregate(self) -> dict:
        names = ("accuracy", "sensitivity", "precision", "recall")
        cm_values = self.confusion.aggregate()
        if not isinstance(cm_values, (tuple, list)):
            cm_values = (cm_values,)
        result = {
            "mean_dice": self._to_jsonable(self.dice.aggregate()),
            "dice_per_class": self._to_jsonable(self.dice.aggregate(reduction="mean_batch")),
            "mean_iou": self._to_jsonable(self.iou.aggregate()),
            "mean_hausdorff": self._to_jsonable(self.hausdorff.aggregate()),
        }
        result.update({name: self._to_jsonable(val) for name, val in zip(names, cm_values)})
        self.reset()
        return result

    def reset(self):
        for metric in (self.dice, self.iou, self.hausdorff, self.confusion):
            metric.reset()


from .math_utils import classwise_dice_numpy
