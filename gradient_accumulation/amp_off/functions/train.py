"""Training utilities with standard and gradient accumulation workflows."""

from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torchvision import models
from tqdm import tqdm

from functions.dataset import COVIDCXNetDataset, DataLoaderConfig, build_transforms, create_dataloader
from functions.evaluation import evaluate_model
from functions.logging import ExperimentLogger, ExperimentLoggerConfig


@dataclass(frozen=True)
class TrainConfig:
    """Hyperparameters and runtime settings for `fit`.

    Attributes:
        num_epochs: Number of training epochs.
        accumulation_steps: Number of mini-steps per optimizer update.
        patience: Early stopping patience on validation loss.
        use_amp: Whether to enable mixed precision on CUDA.
        grad_clip_norm: Optional gradient clipping norm.
        log_every_n_steps: Logging frequency in mini-steps.
        system_log_interval: Interval for system telemetry snapshots.
        run_name: Unique experiment name used for output directory.
        output_dir: Base output directory for logs/checkpoints.
        save_best: Whether to save the best model checkpoint.
        checkpoint_name: Filename for best checkpoint.
        step_scheduler: If True, step scheduler every optimizer update.
        epoch_scheduler: If True, step scheduler at epoch end.
    """

    num_epochs: int = 5
    accumulation_steps: int = 1
    patience: int = 3
    use_amp: bool = True
    grad_clip_norm: Optional[float] = None
    log_every_n_steps: int = 20
    system_log_interval: int = 100
    run_name: str = "experiment"
    output_dir: str = "runs"
    save_best: bool = True
    checkpoint_name: str = "best_model.pth"
    step_scheduler: bool = False
    epoch_scheduler: bool = True


@dataclass
class TrainingResult:
    """Structured outputs of a training run."""

    run_name: str
    run_dir: str
    history: Dict[str, list[float]]
    best_epoch: Optional[int]
    best_val_loss: Optional[float]
    best_checkpoint_path: Optional[str]
    total_train_time_sec: float
    accumulation_steps: int


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """Set random seeds for reproducible experiments.

    Args:
        seed: Random seed value.
        deterministic: If True, enables deterministic CuDNN behavior.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def _sync_if_cuda(device: torch.device) -> None:
    """Synchronize CUDA stream for accurate timing."""

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _peak_vram_mb(device: torch.device) -> Optional[float]:
    """Read current peak GPU memory in MB."""

    if device.type != "cuda":
        return None
    index = device.index if device.index is not None else torch.cuda.current_device()
    return torch.cuda.max_memory_allocated(index) / (1024.0 * 1024.0)


def train_one_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    config: TrainConfig,
    scaler: Optional[GradScaler] = None,
    scheduler: Optional[Any] = None,
    logger: Optional[ExperimentLogger] = None,
    global_step: int = 0,
    optimizer_step_count: int = 0,
    show_progress: bool = True,
) -> Tuple[Dict[str, float], int, int]:
    """Run one training epoch with optional gradient accumulation.

    Args:
        model: Model to train.
        train_loader: Training DataLoader.
        criterion: Loss function.
        optimizer: Optimizer instance.
        device: Active compute device.
        epoch: 1-based epoch index.
        config: Training configuration.
        scaler: AMP grad scaler for CUDA.
        scheduler: Optional learning-rate scheduler.
        logger: Optional experiment logger for CSV/JSON/text outputs.
        global_step: Existing global mini-step counter.
        optimizer_step_count: Existing optimizer-step counter.
        show_progress: Whether to render `tqdm` progress bar.

    Returns:
        Tuple of `(metrics, global_step, optimizer_step_count)`.
    """

    model.train()
    accumulation_steps = max(1, int(config.accumulation_steps))
    amp_enabled = bool(config.use_amp and device.type == "cuda")

    running_loss = 0.0
    running_correct = 0
    total_samples = 0
    total_step_time = 0.0
    total_batches = 0

    optimizer.zero_grad(set_to_none=True)

    progress = train_loader
    if show_progress:
        progress = tqdm(train_loader, desc=f"Epoch {epoch} [train]", leave=False)

    epoch_start = time.perf_counter()
    for batch_idx, (images, labels) in enumerate(progress):
        _sync_if_cuda(device)
        step_start = time.perf_counter()

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(enabled=amp_enabled):
            outputs = model(images)
            loss = criterion(outputs, labels)

        loss_for_backward = loss / accumulation_steps

        if scaler is not None and amp_enabled:
            scaler.scale(loss_for_backward).backward()
        else:
            loss_for_backward.backward()

        should_step = (
            (batch_idx + 1) % accumulation_steps == 0
            or (batch_idx + 1) == len(train_loader)
        )

        if should_step:
            if config.grad_clip_norm is not None:
                if scaler is not None and amp_enabled:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)

            if scaler is not None and amp_enabled:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_step_count += 1

            if scheduler is not None and config.step_scheduler:
                scheduler.step()

        _sync_if_cuda(device)
        step_time = time.perf_counter() - step_start

        batch_size = labels.size(0)
        total_batches += 1
        total_samples += batch_size
        total_step_time += step_time

        batch_loss = float(loss.detach().item())
        preds = outputs.detach().argmax(dim=1)
        correct_batch = (preds == labels).sum().item()
        batch_acc = correct_batch / max(1, batch_size)
        running_loss += batch_loss * batch_size
        running_correct += correct_batch

        global_step += 1
        lr = optimizer.param_groups[0]["lr"]
        samples_per_sec = batch_size / max(step_time, 1e-12)

        if logger is not None and (
            (batch_idx + 1) % max(1, config.log_every_n_steps) == 0
            or (batch_idx + 1) == len(train_loader)
        ):
            logger.log_step(
                {
                    "phase": "train",
                    "epoch": epoch,
                    "global_step": global_step,
                    "optimizer_step": optimizer_step_count,
                    "loss": batch_loss,
                    "accuracy": batch_acc,
                    "step_time_sec": step_time,
                    "samples": batch_size,
                    "batches": 1,
                    "samples_per_sec": samples_per_sec,
                    "batches_per_sec": 1.0 / max(step_time, 1e-12),
                    "lr": lr,
                    "accumulation_steps": accumulation_steps,
                }
            )

        if logger is not None and (
            global_step % max(1, config.system_log_interval) == 0
        ):
            logger.log_system_snapshot(
                device=device,
                phase="train",
                epoch=epoch,
                global_step=global_step,
            )

        if show_progress:
            progress.set_postfix(
                {
                    "loss": f"{batch_loss:.4f}",
                    "acc": f"{batch_acc:.4f}",
                    "sps": f"{samples_per_sec:.1f}",
                }
            )

    _sync_if_cuda(device)
    epoch_time = time.perf_counter() - epoch_start
    avg_loss = running_loss / max(1, total_samples)
    avg_acc = running_correct / max(1, total_samples)
    avg_step_time = total_step_time / max(1, total_batches)

    metrics = {
        "loss": avg_loss,
        "accuracy": avg_acc,
        "epoch_time_sec": epoch_time,
        "avg_step_time_sec": avg_step_time,
        "samples": float(total_samples),
        "batches": float(total_batches),
        "samples_per_sec": total_samples / max(epoch_time, 1e-12),
        "batches_per_sec": total_batches / max(epoch_time, 1e-12),
        "optimizer_steps": float(optimizer_step_count),
    }
    return metrics, global_step, optimizer_step_count


def fit(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: Optional[torch.utils.data.DataLoader],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: TrainConfig,
    scheduler: Optional[Any] = None,
    logger: Optional[ExperimentLogger] = None,
    show_progress: bool = True,
) -> TrainingResult:
    """Train model with optional validation and structured metrics logging.

    Args:
        model: Neural network module.
        train_loader: DataLoader for training data.
        val_loader: Optional validation DataLoader.
        criterion: Loss function.
        optimizer: Optimizer instance.
        device: Active compute device.
        config: Training configuration.
        scheduler: Optional LR scheduler.
        logger: Optional experiment logger. If omitted, one is created.
        show_progress: Whether to display progress bars.

    Returns:
        `TrainingResult` containing history and checkpoint metadata.
    """

    if config.accumulation_steps < 1:
        raise ValueError("accumulation_steps must be >= 1")

    model = model.to(device)
    amp_enabled = bool(config.use_amp and device.type == "cuda")
    scaler = GradScaler(enabled=amp_enabled)

    run_dir = Path(config.output_dir) / config.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    if logger is None:
        logger = ExperimentLogger(
            ExperimentLoggerConfig(run_name=config.run_name, output_dir=config.output_dir)
        )

    best_checkpoint_path = str(run_dir / config.checkpoint_name)
    history: Dict[str, list[float]] = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "epoch_time_sec": [],
        "avg_step_time_sec": [],
        "train_samples_per_sec": [],
        "train_batches_per_sec": [],
        "peak_vram_mb": [],
    }

    best_epoch: Optional[int] = None
    best_val_loss: Optional[float] = None
    epochs_without_improvement = 0
    global_step = 0
    optimizer_step_count = 0

    logger.info(
        f"Run '{config.run_name}' started | epochs={config.num_epochs}, "
        f"accumulation_steps={config.accumulation_steps}, amp={amp_enabled}"
    )

    total_start = time.perf_counter()
    for epoch in range(1, config.num_epochs + 1):
        if device.type == "cuda":
            index = device.index if device.index is not None else torch.cuda.current_device()
            torch.cuda.reset_peak_memory_stats(index)

        train_metrics, global_step, optimizer_step_count = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            config=config,
            scaler=scaler,
            scheduler=scheduler,
            logger=logger,
            global_step=global_step,
            optimizer_step_count=optimizer_step_count,
            show_progress=show_progress,
        )

        val_metrics: Dict[str, float] = {}
        if val_loader is not None:
            val_metrics = evaluate_model(
                model=model,
                data_loader=val_loader,
                criterion=criterion,
                device=device,
                phase="val",
                logger=logger,
                epoch=epoch,
                global_step=global_step,
                show_progress=show_progress,
            )

        peak_vram = _peak_vram_mb(device)
        history["train_loss"].append(train_metrics["loss"])
        history["train_accuracy"].append(train_metrics["accuracy"])
        history["epoch_time_sec"].append(train_metrics["epoch_time_sec"])
        history["avg_step_time_sec"].append(train_metrics["avg_step_time_sec"])
        history["train_samples_per_sec"].append(train_metrics["samples_per_sec"])
        history["train_batches_per_sec"].append(train_metrics["batches_per_sec"])
        history["peak_vram_mb"].append(float(peak_vram) if peak_vram is not None else np.nan)

        if val_loader is not None:
            history["val_loss"].append(val_metrics["loss"])
            history["val_accuracy"].append(val_metrics["accuracy"])
        else:
            history["val_loss"].append(np.nan)
            history["val_accuracy"].append(np.nan)

        logger.log_epoch(
            {
                "phase": "train",
                "epoch": epoch,
                "loss": train_metrics["loss"],
                "accuracy": train_metrics["accuracy"],
                "epoch_time_sec": train_metrics["epoch_time_sec"],
                "avg_step_time_sec": train_metrics["avg_step_time_sec"],
                "samples": int(train_metrics["samples"]),
                "batches": int(train_metrics["batches"]),
                "samples_per_sec": train_metrics["samples_per_sec"],
                "batches_per_sec": train_metrics["batches_per_sec"],
                "optimizer_steps": int(train_metrics["optimizer_steps"]),
                "peak_vram_mb": peak_vram,
            }
        )

        if val_loader is not None:
            logger.log_epoch(
                {
                    "phase": "val",
                    "epoch": epoch,
                    "loss": val_metrics["loss"],
                    "accuracy": val_metrics["accuracy"],
                    "epoch_time_sec": val_metrics["epoch_time_sec"],
                    "avg_step_time_sec": val_metrics["avg_step_time_sec"],
                    "samples": int(val_metrics["samples"]),
                    "batches": int(val_metrics["batches"]),
                    "samples_per_sec": val_metrics["samples_per_sec"],
                    "batches_per_sec": val_metrics["batches_per_sec"],
                    "optimizer_steps": int(train_metrics["optimizer_steps"]),
                    "peak_vram_mb": peak_vram,
                }
            )

        logger.log_system_snapshot(
            device=device,
            phase="epoch_end",
            epoch=epoch,
            global_step=global_step,
        )

        summary = (
            f"Epoch {epoch}/{config.num_epochs} | "
            f"train_loss={train_metrics['loss']:.4f}, train_acc={train_metrics['accuracy']:.4f}"
        )
        if val_loader is not None:
            summary += (
                f", val_loss={val_metrics['loss']:.4f}, val_acc={val_metrics['accuracy']:.4f}"
            )
        summary += (
            f", epoch_time={train_metrics['epoch_time_sec']:.2f}s, "
            f"train_sps={train_metrics['samples_per_sec']:.2f}"
        )
        if peak_vram is not None:
            summary += f", peak_vram={peak_vram:.2f}MB"
        logger.info(summary)

        monitored_loss = val_metrics["loss"] if val_loader is not None else train_metrics["loss"]
        if best_val_loss is None or monitored_loss < best_val_loss:
            best_val_loss = monitored_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            if config.save_best:
                torch.save(model.state_dict(), best_checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= max(1, config.patience):
                logger.info(
                    f"Early stopping triggered at epoch {epoch} "
                    f"(patience={config.patience})."
                )
                break

        if scheduler is not None and config.epoch_scheduler:
            scheduler.step()

    _sync_if_cuda(device)
    total_train_time = time.perf_counter() - total_start

    run_summary = {
        "run_name": config.run_name,
        "num_epochs_completed": len(history["train_loss"]),
        "accumulation_steps": config.accumulation_steps,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "total_train_time_sec": total_train_time,
        "best_checkpoint_path": best_checkpoint_path if config.save_best else None,
    }
    logger.write_summary(run_summary)
    logger.info(f"Run '{config.run_name}' finished in {total_train_time:.2f}s")

    return TrainingResult(
        run_name=config.run_name,
        run_dir=str(run_dir),
        history=history,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        best_checkpoint_path=best_checkpoint_path if config.save_best else None,
        total_train_time_sec=total_train_time,
        accumulation_steps=config.accumulation_steps,
    )


def train(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: Optional[torch.utils.data.DataLoader],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    save_path: str = "best_model.pth",
    num_epochs: int = 5,
    patience: int = 3,
    log_path: str = "training.log",
    accumulation_steps: int = 1,
) -> Tuple[list[float], list[float], list[float], list[float]]:
    """Backward-compatible wrapper around `fit`.

    Returns:
        Tuple of `(train_losses, train_accuracies, val_losses, val_accuracies)`.
    """

    run_name = Path(log_path).stem
    config = TrainConfig(
        num_epochs=num_epochs,
        accumulation_steps=accumulation_steps,
        patience=patience,
        run_name=run_name,
        output_dir=str(Path(log_path).parent or "."),
        checkpoint_name=Path(save_path).name,
    )
    result = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        config=config,
    )
    return (
        result.history["train_loss"],
        result.history["train_accuracy"],
        result.history["val_loss"],
        result.history["val_accuracy"],
    )


def _build_model(model_name: str, num_classes: int) -> nn.Module:
    """Construct torchvision model with updated classification head."""

    if model_name == "resnet18":
        model = models.resnet18(weights="DEFAULT")
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if model_name == "resnet50":
        model = models.resnet50(weights="DEFAULT")
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if model_name == "densenet121":
        model = models.densenet121(weights="DEFAULT")
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        return model
    raise ValueError(f"Unsupported model: {model_name}")


def main() -> None:
    """CLI entrypoint for quick smoke training runs."""

    parser = argparse.ArgumentParser(description="Train AP/PA classifier.")
    parser.add_argument("--data_dir", type=str, required=True, help="Root image directory.")
    parser.add_argument("--csv_file", type=str, required=True, help="Metadata CSV file path.")
    parser.add_argument("--model", type=str, default="resnet18", help="Model architecture.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience.")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of classes.")
    parser.add_argument(
        "--accumulation_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps.",
    )
    parser.add_argument("--run_name", type=str, default="cli_run", help="Experiment name.")
    parser.add_argument("--output_dir", type=str, default="runs", help="Output directory.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    set_seed(args.seed, deterministic=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_transform = build_transforms(image_size=224, augment=True)
    eval_transform = build_transforms(image_size=224, augment=False)

    train_dataset = COVIDCXNetDataset(
        csv_file=args.csv_file,
        root_dir=args.data_dir,
        transform=train_transform,
        split="train",
    )
    val_dataset = None
    for split_name in ("val", "validation", "valid", "test"):
        try:
            val_dataset = COVIDCXNetDataset(
                csv_file=args.csv_file,
                root_dir=args.data_dir,
                transform=eval_transform,
                split=split_name,
            )
            break
        except ValueError:
            continue
    if val_dataset is None:
        raise ValueError(
            "No validation split found. Tried: val, validation, valid, test."
        )

    train_loader = create_dataloader(
        train_dataset,
        DataLoaderConfig(batch_size=args.batch_size, shuffle=True, drop_last=False),
        device=device,
    )
    val_loader = create_dataloader(
        val_dataset,
        DataLoaderConfig(batch_size=args.batch_size, shuffle=False, drop_last=False),
        device=device,
    )

    model = _build_model(args.model, args.num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    config = TrainConfig(
        num_epochs=args.epochs,
        accumulation_steps=args.accumulation_steps,
        patience=args.patience,
        run_name=args.run_name,
        output_dir=args.output_dir,
    )
    fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        config=config,
    )


if __name__ == "__main__":
    main()
