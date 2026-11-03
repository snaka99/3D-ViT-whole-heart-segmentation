import os
import random
from pathlib import Path

import numpy as np
import torch


def random_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_random_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])


def save_training_state(path, model, optimizer, *, fold, step, epoch, batch_offset,
                        limit, interval, dataset_path, config, history, best_dice,
                        accum_loss, accum_n):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "format_version": 1,
        "fold": fold,
        "step": step,
        "epoch": epoch,
        "batch_offset": batch_offset,
        "limit": limit,
        "interval": interval,
        "dataset_path": str(Path(dataset_path).resolve()),
        "config": vars(config).copy(),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": {"step": step, "epoch": epoch},
        "history": history,
        "best_dice": best_dice,
        "accum_loss": accum_loss,
        "accum_n": accum_n,
        "rng_state": random_state(),
    }
    temp_path = path.with_name(path.name + ".tmp")
    try:
        torch.save(state, temp_path)
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def load_training_state(path, model, optimizer, *, fold, limit, interval, dataset_path, config):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("format_version") != 1 or state.get("fold") != fold:
        raise ValueError("Checkpoint format or fold does not match")
    if state["limit"] != limit or state["interval"] != interval:
        raise ValueError("Training iterations/evaluation interval differ from checkpoint")
    if state["config"] != vars(config):
        raise ValueError("YAML configuration differs from checkpoint")
    if state["dataset_path"] != str(Path(dataset_path).resolve()):
        raise ValueError("Dataset path differs from checkpoint")
    model.load_state_dict(state["model_state"])
    optimizer.load_state_dict(state["optimizer_state"])
    return state
