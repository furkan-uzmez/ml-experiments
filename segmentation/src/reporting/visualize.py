from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.reporting.io_utils import load_records_frame
from src.reporting.reporting_contract import (
    CASE_METRICS_FILENAME,
    MODEL_CASE_METRICS_FILENAME,
    RUN_SUMMARY_FILENAME,
    TRAINING_HISTORY_FILENAME,
)

CASE_METRICS_PATH = Path("reports") / CASE_METRICS_FILENAME
MODEL_CASE_METRICS_PATH = Path("reports") / MODEL_CASE_METRICS_FILENAME
RUN_SUMMARY_PATH = Path("reports") / RUN_SUMMARY_FILENAME
UNET_CHECKPOINT_ROOT = Path("artifacts/unet/checkpoints")

TRAINING_LOG_PATTERN = re.compile(
    r"Epoch\s+(?P<epoch>\d+)/\d+\s+\|\s+Train Loss:\s+"
    r"(?P<train_loss>[0-9.]+)\s+\|\s+Val Loss:\s+"
    r"(?P<val_loss>[0-9.]+)\s+\|\s+Val Dice:\s+"
    r"(?P<val_dice>[0-9.]+)"
)


def plot_dice_distribution(model_case_metrics: pd.DataFrame, out_dir: Path) -> None:
    if model_case_metrics.empty:
        return

    plt.figure(figsize=(9, 6))
    ax = sns.violinplot(
        data=model_case_metrics,
        x="model_name",
        y="dice",
        cut=0,
        inner="box",
    )
    ax.set_title("Case-Level Dice Distribution by Model")
    ax.set_xlabel("")
    ax.set_ylabel("Dice")
    ax.set_ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig(out_dir / "dice_distribution.png", dpi=300)
    plt.close()


def plot_boundary_distribution(model_case_metrics: pd.DataFrame, out_dir: Path) -> None:
    if model_case_metrics.empty:
        return

    figure, axes = plt.subplots(1, 2, figsize=(14, 6))
    boundary_specs = (
        ("hd95", "HD95 Distribution", "HD95 (pixels)"),
        ("assd", "ASSD Distribution", "ASSD (pixels)"),
    )

    for axis, (column_name, title, ylabel) in zip(axes, boundary_specs):
        frame = model_case_metrics.dropna(subset=[column_name])
        if frame.empty:
            axis.set_visible(False)
            continue
        sns.boxplot(data=frame, x="model_name", y=column_name, ax=axis)
        axis.set_title(title)
        axis.set_xlabel("")
        axis.set_ylabel(ylabel)

    plt.tight_layout()
    plt.savefig(out_dir / "boundary_distributions.png", dpi=300)
    plt.close()


def plot_lesion_size_scatter(model_case_metrics: pd.DataFrame, out_dir: Path) -> None:
    if model_case_metrics.empty:
        return

    frame = model_case_metrics.copy()
    frame["volume_gt_plot"] = frame["volume_gt"].clip(lower=1.0)

    plt.figure(figsize=(9, 6))
    sns.scatterplot(
        data=frame,
        x="volume_gt_plot",
        y="dice",
        hue="model_name",
        alpha=0.7,
    )
    plt.xscale("log")
    plt.xlabel("Ground-Truth Lesion Size (foreground pixels, log scale)")
    plt.ylabel("Dice")
    plt.title("Lesion Size vs Dice")
    plt.tight_layout()
    plt.savefig(out_dir / "lesion_size_vs_dice.png", dpi=300)
    plt.close()


def sort_runs_for_ranking(run_summary: pd.DataFrame) -> pd.DataFrame:
    return run_summary.assign(
        hd95_rank=run_summary["hd95_mean"].fillna(np.inf),
        assd_rank=run_summary["assd_mean"].fillna(np.inf),
    ).sort_values(
        by=["dice_mean", "hd95_rank", "assd_rank"],
        ascending=[False, True, True],
    )


def choose_representative_case_rows(run_frame: pd.DataFrame) -> list[tuple[str, pd.Series]]:
    ordered = run_frame.sort_values("dice", ascending=True).reset_index(drop=True)
    indices = {
        "Worst": 0,
        "Median": len(ordered) // 2,
        "Best": len(ordered) - 1,
    }
    return [(label, ordered.iloc[index]) for label, index in indices.items()]


def draw_mask_contours(axis: plt.Axes, mask: np.ndarray, color: str) -> None:
    if np.any(mask):
        axis.contour(mask, levels=[0.5], colors=[color], linewidths=1.5)


def plot_overlay_gallery(case_metrics: pd.DataFrame, run_summary: pd.DataFrame, out_dir: Path) -> None:
    if case_metrics.empty or run_summary.empty:
        return

    representative_runs = (
        sort_runs_for_ranking(run_summary)
        .groupby("model_name", sort=False)
        .head(1)
        .reset_index(drop=True)
    )
    model_count = len(representative_runs)
    if model_count == 0:
        return

    figure, axes = plt.subplots(3, model_count, figsize=(4.5 * model_count, 12))
    if model_count == 1:
        axes = np.array(axes).reshape(3, 1)

    for column_index, run_row in representative_runs.iterrows():
        run_frame = case_metrics[case_metrics["run_id"] == run_row["run_id"]]
        if run_frame.empty:
            continue

        for row_index, (label, case_row) in enumerate(choose_representative_case_rows(run_frame)):
            axis = axes[row_index, column_index]
            image = np.array(Image.open(case_row["image_path"]).convert("RGB"))
            ground_truth = np.array(Image.open(case_row["mask_path"]).convert("L")) > 0
            prediction = np.array(Image.open(case_row["prediction_path"]).convert("L")) > 0
            axis.imshow(image)
            draw_mask_contours(axis, ground_truth, "lime")
            draw_mask_contours(axis, prediction, "crimson")
            axis.set_title(
                f"{run_row['model_name']} | {label}\n"
                f"{case_row['case_id']} | Dice {case_row['dice']:.3f}"
            )
            axis.axis("off")

    figure.suptitle("Qualitative Overlay Gallery (green=GT, red=prediction)", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_dir / "overlay_gallery.png", dpi=300)
    plt.close()


def parse_training_log(log_path: Path) -> pd.DataFrame:
    records: list[dict[str, float]] = []
    with open(log_path, "r", encoding="utf-8") as handle:
        for line in handle:
            match = TRAINING_LOG_PATTERN.search(line)
            if match is None:
                continue
            records.append(
                {
                    "epoch": int(match.group("epoch")),
                    "train_loss": float(match.group("train_loss")),
                    "val_loss": float(match.group("val_loss")),
                    "val_dice": float(match.group("val_dice")),
                }
            )
    return pd.DataFrame(records)


def load_unet_training_history() -> pd.DataFrame:
    if not UNET_CHECKPOINT_ROOT.exists():
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    for seed_dir in sorted(UNET_CHECKPOINT_ROOT.glob("seed_*")):
        history_path = seed_dir / TRAINING_HISTORY_FILENAME
        if history_path.exists():
            history_frame = load_records_frame(history_path)
        else:
            log_path = seed_dir / "training_log.log"
            history_frame = parse_training_log(log_path) if log_path.exists() else pd.DataFrame()

        if history_frame.empty:
            continue

        history_frame["seed"] = seed_dir.name.replace("seed_", "")
        frames.append(history_frame)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def plot_learning_curves(out_dir: Path) -> None:
    history = load_unet_training_history()
    if history.empty:
        return

    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.lineplot(data=history, x="epoch", y="train_loss", hue="seed", ax=axes[0], legend=False)
    sns.lineplot(data=history, x="epoch", y="val_loss", hue="seed", ax=axes[0], linestyle="--")
    axes[0].set_title("U-Net Learning Curves")
    axes[0].set_ylabel("Loss")
    axes[0].set_xlabel("Epoch")

    sns.lineplot(data=history, x="epoch", y="val_dice", hue="seed", ax=axes[1])
    axes[1].set_title("U-Net Validation Dice by Seed")
    axes[1].set_ylabel("Val Dice")
    axes[1].set_xlabel("Epoch")

    plt.tight_layout()
    plt.savefig(out_dir / "learning_curves.png", dpi=300)
    plt.close()


def create_visualizations(out_dir: str) -> None:
    case_metrics = load_records_frame(CASE_METRICS_PATH)
    model_case_metrics = load_records_frame(MODEL_CASE_METRICS_PATH)
    run_summary = load_records_frame(RUN_SUMMARY_PATH)

    if case_metrics.empty or model_case_metrics.empty:
        print("Reporting inputs are empty. Skipping visualization.")
        return

    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="colorblind")

    plot_dice_distribution(model_case_metrics, output_dir)
    plot_boundary_distribution(model_case_metrics, output_dir)
    plot_lesion_size_scatter(model_case_metrics, output_dir)
    plot_overlay_gallery(case_metrics, run_summary, output_dir)
    plot_learning_curves(output_dir)
    print(f"Generated visualization plots in {output_dir}")


if __name__ == "__main__":
    create_visualizations("reports/figures")
