"""Evaluation and visualization utilities for classification experiments."""

from __future__ import annotations

import time
from typing import Any, Dict, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from tqdm import tqdm

try:
    from sklearn.metrics import (
        accuracy_score,
        auc,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_curve,
    )
except Exception:  # pragma: no cover - optional dependency
    accuracy_score = None
    auc = None
    confusion_matrix = None
    f1_score = None
    precision_score = None
    recall_score = None
    roc_curve = None


def _sync_if_cuda(device: torch.device) -> None:
    """Synchronize CUDA stream for precise timers."""

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def evaluate_model(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    criterion: Optional[torch.nn.Module],
    device: torch.device,
    phase: str = "val",
    logger: Optional[Any] = None,
    epoch: Optional[int] = None,
    global_step: Optional[int] = None,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """Evaluate a model without gradient tracking and return structured metrics.

    Args:
        model: Trained model.
        data_loader: Validation/test DataLoader.
        criterion: Loss function, or `None` if not needed.
        device: Evaluation device.
        phase: Label for logging (`val` or `test`).
        logger: Optional logger that supports `log_step`.
        epoch: Optional epoch index for log rows.
        global_step: Optional global step value for log rows.
        show_progress: Whether to display progress bars.

    Returns:
        Dictionary with time, throughput, loss/accuracy, and optional advanced metrics.
    """

    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    total_batches = 0
    total_step_time = 0.0

    y_true: list[int] = []
    y_pred: list[int] = []
    y_score: list[float] = []

    progress = data_loader
    if show_progress:
        progress = tqdm(data_loader, desc=f"Evaluate [{phase}]", leave=False)

    _sync_if_cuda(device)
    start_time = time.perf_counter()

    with torch.inference_mode():
        for batch_idx, (images, labels) in enumerate(progress):
            _sync_if_cuda(device)
            step_start = time.perf_counter()

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels) if criterion is not None else None

            probs = F.softmax(outputs, dim=1)
            preds = probs.argmax(dim=1)

            _sync_if_cuda(device)
            step_time = time.perf_counter() - step_start

            batch_size = labels.size(0)
            total_batches += 1
            total_samples += batch_size
            total_step_time += step_time

            if loss is not None:
                total_loss += float(loss.detach().item()) * batch_size
            total_correct += (preds == labels).sum().item()

            labels_cpu = labels.detach().cpu().numpy()
            preds_cpu = preds.detach().cpu().numpy()
            y_true.extend(labels_cpu.tolist())
            y_pred.extend(preds_cpu.tolist())

            if probs.shape[1] > 1:
                y_score.extend(probs[:, 1].detach().cpu().numpy().tolist())

            if logger is not None:
                logger.log_step(
                    {
                        "phase": phase,
                        "epoch": epoch,
                        "global_step": global_step,
                        "optimizer_step": None,
                        "loss": float(loss.item()) if loss is not None else None,
                        "accuracy": (preds == labels).float().mean().item(),
                        "step_time_sec": step_time,
                        "samples": batch_size,
                        "batches": 1,
                        "samples_per_sec": batch_size / max(step_time, 1e-12),
                        "batches_per_sec": 1.0 / max(step_time, 1e-12),
                        "lr": None,
                        "accumulation_steps": 1,
                    }
                )

    _sync_if_cuda(device)
    elapsed = time.perf_counter() - start_time

    avg_loss = total_loss / max(1, total_samples) if criterion is not None else np.nan
    accuracy = total_correct / max(1, total_samples)
    avg_step_time = total_step_time / max(1, total_batches)

    metrics: Dict[str, Any] = {
        "phase": phase,
        "loss": avg_loss,
        "accuracy": accuracy,
        "epoch_time_sec": elapsed,
        "avg_step_time_sec": avg_step_time,
        "samples": float(total_samples),
        "batches": float(total_batches),
        "samples_per_sec": total_samples / max(elapsed, 1e-12),
        "batches_per_sec": total_batches / max(elapsed, 1e-12),
        "y_true": y_true,
        "y_pred": y_pred,
        "y_score": y_score,
    }

    metrics.update(compute_classification_metrics(y_true=y_true, y_pred=y_pred, y_score=y_score))
    return metrics


def compute_classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_score: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """Compute standard classification metrics safely.

    Args:
        y_true: Ground-truth class ids.
        y_pred: Predicted class ids.
        y_score: Optional positive-class probabilities for ROC-AUC.

    Returns:
        Dictionary containing precision/recall/f1/accuracy and ROC info when possible.
    """

    metrics: Dict[str, Any] = {
        "precision": np.nan,
        "recall": np.nan,
        "f1": np.nan,
        "sklearn_accuracy": np.nan,
        "roc_auc": np.nan,
        "fpr": None,
        "tpr": None,
        "confusion_matrix": None,
    }

    if len(y_true) == 0 or len(y_pred) == 0:
        return metrics

    if precision_score is None:
        return metrics

    metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
    metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
    metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)
    metrics["sklearn_accuracy"] = accuracy_score(y_true, y_pred)
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred)

    if y_score is not None and len(y_score) == len(y_true) and len(set(y_true)) > 1:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        metrics["fpr"] = fpr
        metrics["tpr"] = tpr
        metrics["roc_auc"] = auc(fpr, tpr)

    return metrics


def plot_results(
    train_losses: Sequence[float],
    train_accuracies: Sequence[float],
    val_losses: Sequence[float],
    val_accuracies: Sequence[float],
) -> None:
    """Plot train/validation loss and accuracy over epochs."""

    epochs = np.arange(1, len(train_losses) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(epochs, train_losses, marker="o", label="Train Loss")
    axes[0].plot(epochs, val_losses, marker="s", label="Val Loss")
    axes[0].set_title("Loss by Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, train_accuracies, marker="o", label="Train Accuracy")
    axes[1].plot(epochs, val_accuracies, marker="s", label="Val Accuracy")
    axes[1].set_title("Accuracy by Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    plt.show()


def plot_roc_from_metrics(metrics: Mapping[str, Any], title: str = "ROC Curve") -> None:
    """Plot ROC curve from `evaluate_model` output metrics."""

    if metrics.get("fpr") is None or metrics.get("tpr") is None:
        raise ValueError("ROC data unavailable. Ensure y_score exists and both classes are present.")

    fpr = np.asarray(metrics["fpr"])
    tpr = np.asarray(metrics["tpr"])
    roc_auc = metrics.get("roc_auc", np.nan)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right")
    plt.show()


def plot_confusion_from_metrics(
    metrics: Mapping[str, Any],
    class_names: Sequence[str] = ("AP", "PA"),
    title: str = "Confusion Matrix",
) -> None:
    """Plot confusion matrix from `evaluate_model` output metrics."""

    cm = metrics.get("confusion_matrix")
    if cm is None:
        raise ValueError("Confusion matrix data unavailable.")

    cm_array = np.asarray(cm)
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm_array,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def eval_on_metrics(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    criterion: Optional[torch.nn.Module] = None,
    device: Optional[torch.device] = None,
    show_plots: bool = True,
) -> Dict[str, Any]:
    """Backward-compatible test evaluation helper.

    Args:
        model: Trained model.
        test_loader: Test DataLoader.
        criterion: Optional loss function.
        device: Optional device override.
        show_plots: Whether to render ROC and confusion matrix when available.

    Returns:
        Dictionary of test metrics.
    """

    if device is None:
        device = next(model.parameters()).device

    metrics = evaluate_model(
        model=model,
        data_loader=test_loader,
        criterion=criterion,
        device=device,
        phase="test",
        logger=None,
        epoch=None,
        global_step=None,
        show_progress=True,
    )

    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    if not np.isnan(metrics["precision"]):
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1 Score:  {metrics['f1']:.4f}")
    if not np.isnan(metrics["roc_auc"]):
        print(f"AUC:       {metrics['roc_auc']:.4f}")

    if show_plots:
        if metrics.get("fpr") is not None:
            plot_roc_from_metrics(metrics, title="ROC on Test Data")
        if metrics.get("confusion_matrix") is not None:
            plot_confusion_from_metrics(metrics, title="Confusion Matrix on Test Data")

    return metrics
