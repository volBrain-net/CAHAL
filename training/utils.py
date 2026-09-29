import numpy as np
import torch
from torch import optim


def configure_device():
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("No GPU available.")
    else:
        torch.set_float32_matmul_precision("high")
    return device


class EarlyStopping:
    def __init__(self, patience=100, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


class ModelCheckpoint:
    def __init__(self, filepath, mode="min"):
        self.filepath = filepath
        self.mode = mode
        self.best = float("inf") if mode == "min" else float("-inf")

    def save_checkpoint(self, model, value):
        improved = value < self.best if self.mode == "min" else value > self.best
        if improved:
            self.best = value
            torch.save(model.state_dict(), self.filepath)


class SequentialPlateauScheduler:
    """
    Chains several ReduceLROnPlateau schedulers. When one bottoms out at its
    min_lr, switches to the next and resets the LR to its initial value
    (used to escape local minima during long training runs).
    """

    def __init__(self, optimizer, schedulers_configs, initial_lr=None):
        self.optimizer = optimizer
        self.schedulers_configs = schedulers_configs
        self.current_idx = 0
        self.initial_lr = initial_lr if initial_lr is not None else optimizer.param_groups[0]["lr"]
        self.schedulers = [
            optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode=c.get("mode", "min"), factor=c.get("factor", 0.5),
                patience=c.get("patience", 10), min_lr=c.get("min_lr", 0),
            )
            for c in schedulers_configs
        ]

    def step(self, metrics):
        self.schedulers[self.current_idx].step(metrics)
        new_lr = self.optimizer.param_groups[0]["lr"]
        min_lr = self.schedulers_configs[self.current_idx].get("min_lr", 0)

        if new_lr <= min_lr and self.current_idx < len(self.schedulers) - 1:
            self.current_idx += 1
            for group in self.optimizer.param_groups:
                group["lr"] = self.initial_lr
            next_scheduler = self.schedulers[self.current_idx]
            next_scheduler.best = metrics
            next_scheduler.num_bad_epochs = 0


class Segmentation_model:
    """
    Wraps a frozen, pretrained segmentation network used as an auxiliary
    supervision signal during reconstruction training. Provide the
    architecture (already instantiated) and the path to its weights.
    """

    def __init__(self, model_path, model_architecture, device):
        self.model = model_architecture
        self.device = device
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.to(device)

    def __call__(self, x):
        """Test-time augmentation: average predictions on x and its flip along dim 2."""
        with torch.no_grad():
            pred = self.model(x)
            pred_flipped = torch.flip(self.model(torch.flip(x, [2])), [2])
        return (pred + pred_flipped) / 2


class DataGenerationLogger:
    """Accumulates per-sample LR resolution/anisotropy stats for TensorBoard logging."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.resolutions = []

    def add_sample(self, info):
        if isinstance(info, list):
            for single in info:
                self.resolutions.append(single["resolutions"])
        else:
            self.resolutions.append(info["resolutions"])

    def log_to_tensorboard(self, writer, epoch, prefix="Train"):
        if not self.resolutions:
            return
        res = np.array(self.resolutions)
        product = res.prod(axis=1)
        anisotropy = res.min(axis=1) / res.max(axis=1)
        writer.add_scalar(f"{prefix}/ResProduct_Mean", product.mean(), epoch)
        writer.add_scalar(f"{prefix}/Anisotropy_Mean", anisotropy.mean(), epoch)
