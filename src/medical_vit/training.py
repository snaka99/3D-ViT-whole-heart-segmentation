import json
import logging
import math
from pathlib import Path
import pickle

import torch
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.utils import set_determinism
from tqdm import tqdm

from .config import ExperimentConfig
from .data import make_cv_loaders
from .metrics import MetricCollector
from .model import ViTSegmentation
from .math_utils import warmup_exponential_lr
from .checkpoints import (load_training_state, save_training_state, restore_random_state)

LOGGER = logging.getLogger(__name__)


def validate(model, val_loader, device: torch.device, config: ExperimentConfig) -> dict:
    model.eval()
    collector = MetricCollector(config)
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation", leave=False):
            x = batch["image"].to(device)
            y = batch["label"].to(device)
            logits = sliding_window_inference(x, config.img_size, config.sw_batch_size, model)
            collector.update(logits, y)
    return collector.aggregate()


def run_training(
    config: ExperimentConfig,
    dataset_path: Path,
    out_dir: Path,
    device: torch.device,
    num_workers: int = 0,
    max_iterations: int | None = None,
    eval_interval: int | None = None,
    resume: bool = False,
    checkpoint_interval: int = 500,
) -> list[dict]:
    config.validate()
    set_determinism(seed=config.train_seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    limit = max_iterations or config.max_iterations
    interval = eval_interval or config.eval_interval
    if checkpoint_interval <= 0:
        raise ValueError("checkpoint interval must be positive")
    if limit <= 0 or interval <= 0:
        raise ValueError("iterations and eval interval must be positive")
    all_folds = []
    for fold, train_loader, val_loader in make_cv_loaders(
        config, dataset_path, num_workers=num_workers
    ):
        LOGGER.info("Fold %s of %s", fold, config.nfolds)
        fold_dir = out_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        model = ViTSegmentation(
            in_channels=config.image_channels, img_size=config.img_size,
            patch_size=config.patch_size, num_classes=config.label_classes,
        ).to(device)
        loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, amsgrad=True)
        max_epochs = max(config.warmup_epochs + 1, math.ceil(limit / len(train_loader)))
        history = {"fold": fold, "loss_values": [], "mean_dice_values": [],
                   "evaluation_steps": [], "best": None, "best_step": None}
        best_dice = float("-inf")
        step = 0
        epoch = 0
        batch_offset = 0
        accum_loss = 0.0
        accum_n = 0
        resume_state = None
        checkpoint_path = fold_dir / "training_state.pth"
        metrics_path = fold_dir / "metrics.json"
        last_path = fold_dir / "last_model.pth"

        if resume and metrics_path.exists() and last_path.exists():
            previous = json.loads(metrics_path.read_text(encoding="utf-8"))
            if previous.get("evaluation_steps", [None])[-1] == limit:
                LOGGER.info("Fold %d completed; skipping", fold)
                all_folds.append(previous)
                continue

        if resume and checkpoint_path.exists():
            resume_state = load_training_state(
                checkpoint_path, model, optimizer, fold=fold, limit=limit,
                interval=interval, dataset_path=dataset_path, config=config,
            )
            step = resume_state["step"]
            epoch = resume_state["epoch"]
            batch_offset = resume_state["batch_offset"]
            history = resume_state["history"]
            best_dice = resume_state["best_dice"]
            accum_loss = resume_state["accum_loss"]
            accum_n = resume_state["accum_n"]
            LOGGER.info("Fold %d resuming from step %d", fold, step)
        elif resume and (fold_dir / "best_model.pth").exists():
            LOGGER.warning("Fold %d has legacy weights only; starting this fold again", fold)
        elif not resume and any(fold_dir.iterdir()):
            raise FileExistsError(
                f"Output folder {fold_dir} contains files. Use --resume or another --output."
            )

        while step < limit:
            # Deterministic sample order for each fold and epoch.
            if hasattr(train_loader.sampler, "generator"):
                train_loader.sampler.generator = torch.Generator().manual_seed(
                    config.train_seed + fold * 100000 + epoch
                )
            if resume_state is not None:
                restore_random_state(resume_state["rng_state"])
            iterator = iter(train_loader)
            for _ in range(batch_offset):
                next(iterator)
            if resume_state is not None:
                restore_random_state(resume_state["rng_state"])
                resume_state = None
            model.train()
            for index in tqdm(range(batch_offset, len(train_loader)),
                              desc=f"Fold {fold} train", leave=False):
                batch = next(iterator)
                x = batch["image"].to(device)
                y = batch["label"].to(device)
                logits = model(x)
                loss = loss_fn(logits, y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                next_lr = warmup_exponential_lr(
                    config.lr, (step // len(train_loader)), config.warmup_epochs, max_epochs
                )
                for param_group in optimizer.param_groups:
                    param_group["lr"] = next_lr
                step += 1
                batch_offset = index + 1
                accum_loss += float(loss.item())
                accum_n += 1
                if step % interval == 0 or step == limit:
                    metrics = validate(model, val_loader, device, config)
                    history["loss_values"].append(accum_loss / max(1, accum_n))
                    history["mean_dice_values"].append(metrics["mean_dice"])
                    history["evaluation_steps"].append(step)
                    accum_loss, accum_n = 0.0, 0
                    dice = metrics["mean_dice"]
                    if dice is not None and not math.isnan(dice) and dice > best_dice:
                        best_dice = dice
                        history["best"] = metrics
                        history["best_step"] = step
                        torch.save(model.state_dict(), fold_dir / "best_model.pth")
                        LOGGER.info("Fold %d step %d: best Dice %.5f", fold, step, dice)
                    model.train()
                if step % checkpoint_interval == 0 or step == limit:
                    save_training_state(
                        checkpoint_path, model, optimizer, fold=fold, step=step,
                        epoch=epoch, batch_offset=batch_offset, limit=limit,
                        interval=interval, dataset_path=dataset_path, config=config,
                        history=history, best_dice=best_dice,
                        accum_loss=accum_loss, accum_n=accum_n,
                    )
                if step >= limit:
                    break
            epoch += 1
            batch_offset = 0
            resume = bool(resume)
        torch.save(model.state_dict(), fold_dir / "last_model.pth")
        (fold_dir / "metrics.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        with (fold_dir / "metrics.pkl").open("wb") as fh:
            pickle.dump(history, fh)
        all_folds.append(history)
    (out_dir / "summary.json").write_text(json.dumps(all_folds, indent=2), encoding="utf-8")
    return all_folds
