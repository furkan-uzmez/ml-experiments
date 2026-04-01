import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.reporting.aggregate_metrics import (
    build_model_case_metrics,
    build_model_summary,
    build_run_summary,
)
from src.reporting.generate_report import generate_markdown_report
from src.reporting.io_utils import write_json


def test_run_and_model_summary_include_boundary_metrics():
    case_metrics = pd.DataFrame(
        [
            {
                "run_id": "unet_seed_11",
                "experiment_group": "benchmark",
                "model_name": "U-Net",
                "backbone": "BasicUNet",
                "task_type": "binary_segmentation",
                "dataset_version": "ISIC2018_Task1",
                "split_version": "primary_split",
                "label_version": "ISIC2018_Task1_binary_segmentation_labels",
                "seed": 11,
                "loss_name": "DiceFocalLoss",
                "optimizer": "AdamW",
                "learning_rate": 1e-4,
                "scheduler": "none",
                "augmentation_summary": "{}",
                "preprocessing_summary": "{}",
                "threshold_policy": "fixed_threshold@0.5",
                "threshold_used": 0.5,
                "checkpoint_path": "best_model.pth",
                "train_started_at": "2026-04-01T10:00:00Z",
                "train_finished_at": "2026-04-01T10:10:00Z",
                "split": "test",
                "case_id": "ISIC_1",
                "patient_id": "ISIC_1",
                "class_name": "lesion",
                "dice": 0.9,
                "iou": 0.82,
                "hd95": 3.0,
                "assd": 0.8,
                "lesion_count_gt": 1,
                "lesion_count_pred": 1,
                "volume_gt": 200.0,
                "volume_pred": 195.0,
                "inference_time_seconds": 0.12,
                "peak_gpu_memory_mb": 512.0,
                "image_path": "image_1.jpg",
                "mask_path": "mask_1.png",
                "prediction_path": "pred_1.png",
            },
            {
                "run_id": "unet_seed_11",
                "experiment_group": "benchmark",
                "model_name": "U-Net",
                "backbone": "BasicUNet",
                "task_type": "binary_segmentation",
                "dataset_version": "ISIC2018_Task1",
                "split_version": "primary_split",
                "label_version": "ISIC2018_Task1_binary_segmentation_labels",
                "seed": 11,
                "loss_name": "DiceFocalLoss",
                "optimizer": "AdamW",
                "learning_rate": 1e-4,
                "scheduler": "none",
                "augmentation_summary": "{}",
                "preprocessing_summary": "{}",
                "threshold_policy": "fixed_threshold@0.5",
                "threshold_used": 0.5,
                "checkpoint_path": "best_model.pth",
                "train_started_at": "2026-04-01T10:00:00Z",
                "train_finished_at": "2026-04-01T10:10:00Z",
                "split": "test",
                "case_id": "ISIC_2",
                "patient_id": "ISIC_2",
                "class_name": "lesion",
                "dice": 0.7,
                "iou": 0.56,
                "hd95": 6.0,
                "assd": 1.5,
                "lesion_count_gt": 1,
                "lesion_count_pred": 1,
                "volume_gt": 100.0,
                "volume_pred": 120.0,
                "inference_time_seconds": 0.15,
                "peak_gpu_memory_mb": 520.0,
                "image_path": "image_2.jpg",
                "mask_path": "mask_2.png",
                "prediction_path": "pred_2.png",
            },
        ]
    )

    run_summary = build_run_summary(case_metrics)
    model_case_metrics = build_model_case_metrics(case_metrics)
    model_summary = build_model_summary(model_case_metrics, run_summary)

    assert "assd_mean" in run_summary.columns
    assert "assd_mean" in model_summary.columns
    assert run_summary.iloc[0]["cases_evaluated"] == 2
    assert model_summary.iloc[0]["dice_mean"] == case_metrics["dice"].mean()


def test_generate_markdown_report_contains_required_sections(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs").mkdir()
    (tmp_path / "splits").mkdir()
    (tmp_path / "reports").mkdir()

    (tmp_path / "configs" / "dataset.yaml").write_text(
        "\n".join(
            [
                "dataset:",
                '  name: "ISIC2018_Task1"',
                '  task_type: "binary_segmentation"',
                '  split_file: "splits/primary_split.json"',
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "splits" / "primary_split.json").write_text(
        json.dumps({"train": ["A"], "val": ["B"], "test": ["C", "D"]}),
        encoding="utf-8",
    )

    write_json(
        tmp_path / "reports" / "case_metrics.json",
        [
            {
                "run_id": "unet_seed_11",
                "case_id": "C",
                "dice": 0.9,
                "hd95": 3.0,
                "assd": 0.8,
            },
            {
                "run_id": "unet_seed_11",
                "case_id": "D",
                "dice": 0.8,
                "hd95": 4.0,
                "assd": 1.0,
            },
        ],
    )

    write_json(
        tmp_path / "reports" / "model_case_metrics.json",
        [
            {
                "model_name": "U-Net",
                "split": "test",
                "case_id": "C",
                "patient_id": "C",
                "class_name": "lesion",
                "image_path": "image_C.jpg",
                "mask_path": "mask_C.png",
                "prediction_path": "pred_C.png",
                "dice": 0.9,
                "iou": 0.82,
                "hd95": 3.0,
                "assd": 0.8,
                "lesion_count_gt": 1,
                "lesion_count_pred": 1,
                "volume_gt": 200.0,
                "volume_pred": 190.0,
                "inference_time_seconds": 0.12,
                "peak_gpu_memory_mb": 512.0,
            },
            {
                "model_name": "U-Net",
                "split": "test",
                "case_id": "D",
                "patient_id": "D",
                "class_name": "lesion",
                "image_path": "image_D.jpg",
                "mask_path": "mask_D.png",
                "prediction_path": "pred_D.png",
                "dice": 0.8,
                "iou": 0.67,
                "hd95": 4.0,
                "assd": 1.0,
                "lesion_count_gt": 1,
                "lesion_count_pred": 1,
                "volume_gt": 100.0,
                "volume_pred": 110.0,
                "inference_time_seconds": 0.13,
                "peak_gpu_memory_mb": 510.0,
            },
        ],
    )

    write_json(
        tmp_path / "reports" / "run_summary.json",
        [
            {
                "run_id": "unet_seed_11",
                "model_name": "U-Net",
                "seed": 11,
                "loss_name": "DiceFocalLoss",
                "threshold_policy": "fixed_threshold@0.5",
                "dice_mean": 0.85,
                "hd95_mean": 3.5,
                "assd_mean": 0.9,
                "inference_time_seconds_mean": 0.125,
            }
        ],
    )

    write_json(
        tmp_path / "reports" / "model_summary.json",
        [
            {
                "model_name": "U-Net",
                "num_runs": 1,
                "dice_mean": 0.85,
                "dice_std": 0.05,
                "iou_mean": 0.745,
                "iou_std": 0.075,
                "hd95_mean": 3.5,
                "hd95_std": 0.5,
                "assd_mean": 0.9,
                "assd_std": 0.1,
                "inference_time_seconds_mean": 0.125,
                "peak_gpu_memory_mb_max": 512.0,
            }
        ],
    )

    write_json(
        tmp_path / "reports" / "run_inventory.json",
        [
            {
                "model_name": "U-Net",
                "threshold_policy": "fixed_threshold@0.5",
                "label_version": "ISIC2018_Task1_binary_segmentation_labels",
                "split_version": "primary_split",
            }
        ],
    )

    generate_markdown_report("reports")

    report_path = tmp_path / "reports" / "BENCHMARK_REPORT.md"
    report = report_path.read_text(encoding="utf-8")
    assert "## 1. Task Fingerprint" in report
    assert "## 4. Threshold And Calibration Analysis" in report
    assert "## 7. Recommendation" in report
    assert "probability-based calibration analysis is unavailable" in report
