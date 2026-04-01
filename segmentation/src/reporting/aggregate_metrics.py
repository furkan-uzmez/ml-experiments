from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import SimpleITK as sitk
import yaml
from scipy import ndimage
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.dataio.dataset_index import DatasetIndex
from src.dataio.split_loader import SplitLoader
from src.evaluation.metrics import compute_metrics
from src.evaluation.resample import resample_to_reference
from src.reporting.io_utils import (
    load_json,
    load_jsonl_log,
    write_json,
    write_records_frame,
)
from src.reporting.reporting_contract import (
    CASE_MEASUREMENT_COLUMNS,
    CASE_METRICS_FILENAME,
    MODEL_CASE_METRICS_FILENAME,
    MODEL_SUMMARY_FILENAME,
    RUN_INVENTORY_FILENAME,
    RUNTIME_LOG_FILENAME,
    RUN_METADATA_COLUMNS,
    RUN_SUMMARY_FILENAME,
    RunSpec,
    SEGMENTATION_CLASS_NAME,
)

REPORTS_DIR = Path("reports")
EXPERIMENT_GROUP = "isic2018_segmentation_benchmark"
SUMMARY_METRICS = ("dice", "iou", "hd95", "assd")


def read_yaml(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def compact_config_summary(config: dict[str, Any] | None) -> str:
    if not config:
        return "not_recorded"
    return json.dumps(config, sort_keys=True, separators=(",", ":"))


def safe_json_load(path: Path) -> dict[str, Any]:
    return load_json(path, {})


def infer_label_version(dataset_config: dict[str, Any]) -> str:
    dataset_name = dataset_config.get("name", "dataset")
    task_type = dataset_config.get("task_type", "segmentation")
    return f"{dataset_name}_{task_type}_labels"


def build_run_specs() -> list[RunSpec]:
    dataset_config = read_yaml("configs/dataset.yaml")["dataset"]
    seeds_config = read_yaml("configs/seeds.yaml")["seeds"]
    unet_config = read_yaml("configs/unet.yaml")
    medsam_config = read_yaml("configs/medsam.yaml")

    dataset_version = dataset_config.get("name", "dataset")
    split_version = Path(dataset_config.get("split_file", "splits/primary_split.json")).stem
    label_version = infer_label_version(dataset_config)
    task_type = dataset_config.get("task_type", "binary_segmentation")
    training_seeds = seeds_config.get("training_runs", [11, 22, 33])

    run_specs: list[RunSpec] = []
    unet_checkpoint_root = Path(unet_config["training"]["save_dir"])
    unet_prediction_root = Path(unet_config["inference"]["output_dir"])

    for seed in training_seeds:
        checkpoint_dir = unet_checkpoint_root / f"seed_{seed}"
        manifest = safe_json_load(checkpoint_dir / "run_manifest.json")
        run_specs.append(
            RunSpec(
                run_id=f"unet_seed_{seed}",
                experiment_group=EXPERIMENT_GROUP,
                model_name="U-Net",
                backbone=unet_config["model"].get("architecture", "BasicUNet"),
                task_type=task_type,
                dataset_version=dataset_version,
                split_version=split_version,
                label_version=label_version,
                seed=seed,
                loss_name=manifest.get("loss_name", unet_config["loss"].get("name", "DiceFocalLoss")),
                optimizer=manifest.get("optimizer", unet_config.get("optimizer", {}).get("name", "AdamW")),
                learning_rate=manifest.get("learning_rate", unet_config["training"].get("learning_rate")),
                scheduler=manifest.get("scheduler", "none"),
                augmentation_summary=manifest.get(
                    "augmentation_summary",
                    compact_config_summary(unet_config.get("augmentation")),
                ),
                preprocessing_summary=manifest.get(
                    "preprocessing_summary",
                    compact_config_summary(unet_config.get("preprocessing")),
                ),
                threshold_policy=manifest.get("threshold_policy", "fixed_threshold@0.5"),
                threshold_used=0.5,
                checkpoint_path=str(checkpoint_dir / "best_model.pth"),
                prediction_dir=str(unet_prediction_root / f"seed_{seed}"),
                prediction_suffix="_segmentation.png",
                runtime_log_path=str(
                    unet_prediction_root / f"seed_{seed}" / RUNTIME_LOG_FILENAME
                ),
                train_started_at=manifest.get("train_started_at"),
                train_finished_at=manifest.get("train_finished_at"),
            )
        )

    for seed in training_seeds:
        nnunet_prediction_dir = (
            Path("artifacts/nnunet/nnUNet_results/Dataset500_ISIC2018")
            / f"predictions_seed_{seed}"
        )
        run_specs.append(
            RunSpec(
                run_id=f"nnunet_seed_{seed}",
                experiment_group=EXPERIMENT_GROUP,
                model_name="nnU-Net",
                backbone="nnUNet v2 2D",
                task_type=task_type,
                dataset_version=dataset_version,
                split_version=split_version,
                label_version=label_version,
                seed=seed,
                loss_name="nnUNet default",
                optimizer="nnUNet default",
                learning_rate=None,
                scheduler="nnUNet default",
                augmentation_summary="nnUNet self-configuring augmentation",
                preprocessing_summary="nnUNet self-configuring preprocessing",
                threshold_policy="hard-mask export",
                threshold_used=None,
                checkpoint_path=str(
                    Path("artifacts/nnunet/nnUNet_results/Dataset500_ISIC2018")
                ),
                prediction_dir=str(nnunet_prediction_dir),
                prediction_suffix=".png",
                runtime_log_path=str(nnunet_prediction_dir / RUNTIME_LOG_FILENAME),
            )
        )

    medsam_mode = medsam_config.get("mode", "zero_shot")
    medsam_output_dir = Path(medsam_config["inference"]["output_dir"])
    medsam_checkpoint = (
        Path(medsam_config["model"]["checkpoint_dir"])
        / medsam_config["model"]["checkpoint_name"]
    )
    medsam_backbone = f"MedSAM {medsam_config['model'].get('type', 'vit_b')}"
    medsam_preprocessing = "MedSAM encoder preprocessing at 1024x1024"
    medsam_threshold_policy = "decoder hard mask; probabilities not exported"

    if medsam_mode == "fine_tuned":
        for seed in training_seeds:
            prediction_dir = medsam_output_dir / f"seed_{seed}"
            run_specs.append(
                RunSpec(
                    run_id=f"medsam_seed_{seed}",
                    experiment_group=EXPERIMENT_GROUP,
                    model_name="MedSAM",
                    backbone=medsam_backbone,
                    task_type=task_type,
                    dataset_version=dataset_version,
                    split_version=split_version,
                    label_version=label_version,
                    seed=seed,
                    loss_name="task-specific fine-tuning",
                    optimizer="not_recorded",
                    learning_rate=None,
                    scheduler="not_recorded",
                    augmentation_summary="not_recorded",
                    preprocessing_summary=medsam_preprocessing,
                    threshold_policy=medsam_threshold_policy,
                    threshold_used=None,
                    checkpoint_path=str(medsam_checkpoint),
                    prediction_dir=str(prediction_dir),
                    prediction_suffix="_pred.png",
                    runtime_log_path=str(prediction_dir / RUNTIME_LOG_FILENAME),
                )
            )
    else:
        run_specs.append(
            RunSpec(
                run_id="medsam_zero_shot",
                experiment_group=EXPERIMENT_GROUP,
                model_name="MedSAM",
                backbone=medsam_backbone,
                task_type=task_type,
                dataset_version=dataset_version,
                split_version=split_version,
                label_version=label_version,
                seed=0,
                loss_name="not_applicable_zero_shot",
                optimizer="not_applicable_zero_shot",
                learning_rate=None,
                scheduler="not_applicable_zero_shot",
                augmentation_summary="not_applicable_zero_shot",
                preprocessing_summary=medsam_preprocessing,
                threshold_policy=medsam_threshold_policy,
                threshold_used=None,
                checkpoint_path=str(medsam_checkpoint),
                prediction_dir=str(medsam_output_dir),
                prediction_suffix="_pred.png",
                runtime_log_path=str(medsam_output_dir / RUNTIME_LOG_FILENAME),
            )
        )

    return run_specs


def load_runtime_lookup(runtime_path: str | None) -> dict[str, dict[str, float]]:
    if runtime_path is None:
        return {}

    runtime_file = Path(runtime_path)
    if not runtime_file.exists():
        return {}

    runtime_records = load_jsonl_log(runtime_file)
    lookup: dict[str, dict[str, float]] = {}
    for row in runtime_records:
        case_id = row.get("case_id")
        if not case_id:
            continue
        lookup[str(case_id)] = {
            "inference_time_seconds": float(row.get("inference_time_seconds", np.nan)),
            "peak_gpu_memory_mb": float(row.get("peak_gpu_memory_mb", np.nan)),
        }
    return lookup


def count_lesions(mask: np.ndarray) -> int:
    _, component_count = ndimage.label(mask.astype(np.uint8))
    return int(component_count)


def count_foreground_pixels(mask: np.ndarray) -> float:
    return float(np.count_nonzero(mask))


def evaluate_run(
    run_spec: RunSpec,
    test_ids: list[str],
    dataset_index: DatasetIndex,
) -> pd.DataFrame:
    prediction_dir = Path(run_spec.prediction_dir)
    if not prediction_dir.exists():
        print(f"Warning: Directory not found - {prediction_dir}. Skipping {run_spec.run_id}.")
        return pd.DataFrame()

    runtime_lookup = load_runtime_lookup(run_spec.runtime_log_path)
    records: list[dict[str, Any]] = []

    print(f"Aggregating {run_spec.run_id} from {prediction_dir}...")
    for case_id in tqdm(test_ids, desc=run_spec.run_id):
        prediction_path = prediction_dir / f"{case_id}{run_spec.prediction_suffix}"
        if not prediction_path.exists():
            continue

        case_paths = dataset_index.get_case(case_id)
        ground_truth_path = case_paths["mask_path"]

        try:
            prediction_image = sitk.ReadImage(str(prediction_path))
            ground_truth_image = sitk.ReadImage(ground_truth_path)
            prediction_mask = resample_to_reference(
                prediction_image,
                ground_truth_image,
                is_binary=True,
            )
            ground_truth_mask = sitk.GetArrayFromImage(ground_truth_image)
            prediction_mask = (prediction_mask > 0).astype(np.uint8)
            ground_truth_mask = (ground_truth_mask > 0).astype(np.uint8)
            metrics = compute_metrics(prediction_mask, ground_truth_mask)
        except Exception as error:  # pragma: no cover - defensive IO boundary
            print(f"Failed on {case_id} for {run_spec.run_id}: {error}")
            continue

        runtime_metrics = runtime_lookup.get(case_id, {})
        record = run_spec.inventory_record()
        record.update(
            {
                "split": "test",
                "case_id": case_id,
                "patient_id": case_id,
                "class_name": SEGMENTATION_CLASS_NAME,
                "dice": metrics["dice"],
                "iou": metrics["iou"],
                "hd95": metrics["hd95"],
                "assd": metrics["assd"],
                "lesion_count_gt": count_lesions(ground_truth_mask),
                "lesion_count_pred": count_lesions(prediction_mask),
                "volume_gt": count_foreground_pixels(ground_truth_mask),
                "volume_pred": count_foreground_pixels(prediction_mask),
                "inference_time_seconds": runtime_metrics.get(
                    "inference_time_seconds",
                    np.nan,
                ),
                "peak_gpu_memory_mb": runtime_metrics.get("peak_gpu_memory_mb", np.nan),
                "image_path": case_paths["image_path"],
                "mask_path": case_paths["mask_path"],
                "prediction_path": str(prediction_path),
            }
        )
        records.append(record)

    return pd.DataFrame(records)


def summarize_series(series: pd.Series) -> dict[str, float]:
    clean = series.dropna()
    if clean.empty:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "median": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }

    return {
        "mean": float(clean.mean()),
        "std": float(clean.std(ddof=0)),
        "median": float(clean.median()),
        "min": float(clean.min()),
        "max": float(clean.max()),
    }


def build_run_summary(case_metrics: pd.DataFrame) -> pd.DataFrame:
    if case_metrics.empty:
        return pd.DataFrame()

    summary_rows: list[dict[str, Any]] = []
    for run_id, run_frame in case_metrics.groupby("run_id", sort=False):
        summary_row = {
            column: run_frame.iloc[0][column]
            for column in RUN_METADATA_COLUMNS
        }
        summary_row["cases_evaluated"] = int(len(run_frame))

        for metric_name in SUMMARY_METRICS:
            summary = summarize_series(run_frame[metric_name])
            for statistic_name, value in summary.items():
                summary_row[f"{metric_name}_{statistic_name}"] = value

        runtime_mean = summarize_series(run_frame["inference_time_seconds"])["mean"]
        memory_summary = summarize_series(run_frame["peak_gpu_memory_mb"])
        summary_row["inference_time_seconds_mean"] = runtime_mean
        summary_row["peak_gpu_memory_mb_mean"] = memory_summary["mean"]
        summary_row["peak_gpu_memory_mb_max"] = memory_summary["max"]
        summary_rows.append(summary_row)

    return pd.DataFrame(summary_rows)


def build_model_case_metrics(case_metrics: pd.DataFrame) -> pd.DataFrame:
    if case_metrics.empty:
        return pd.DataFrame()

    group_columns = [
        "model_name",
        "split",
        "case_id",
        "patient_id",
        "class_name",
        "image_path",
        "mask_path",
    ]
    numeric_columns = list(CASE_MEASUREMENT_COLUMNS)
    aggregated = (
        case_metrics.groupby(group_columns, as_index=False)[numeric_columns]
        .mean(numeric_only=True)
    )
    aggregated["prediction_path"] = ""
    return aggregated


def build_model_summary(
    model_case_metrics: pd.DataFrame,
    run_summary: pd.DataFrame,
) -> pd.DataFrame:
    if model_case_metrics.empty:
        return pd.DataFrame()

    summary_rows: list[dict[str, Any]] = []
    for model_name, model_frame in model_case_metrics.groupby("model_name", sort=False):
        model_runs = run_summary[run_summary["model_name"] == model_name]
        summary_row: dict[str, Any] = {
            "model_name": model_name,
            "num_runs": int(len(model_runs)),
            "cases_evaluated": int(len(model_frame)),
        }

        for metric_name in SUMMARY_METRICS:
            summary = summarize_series(model_frame[metric_name])
            for statistic_name, value in summary.items():
                summary_row[f"{metric_name}_{statistic_name}"] = value

        summary_row["run_dice_mean"] = summarize_series(model_runs["dice_mean"])["mean"]
        summary_row["run_dice_std"] = summarize_series(model_runs["dice_mean"])["std"]
        summary_row["run_hd95_mean"] = summarize_series(model_runs["hd95_mean"])["mean"]
        summary_row["run_hd95_std"] = summarize_series(model_runs["hd95_mean"])["std"]
        summary_row["run_assd_mean"] = summarize_series(model_runs["assd_mean"])["mean"]
        summary_row["run_assd_std"] = summarize_series(model_runs["assd_mean"])["std"]
        summary_row["inference_time_seconds_mean"] = summarize_series(
            model_runs["inference_time_seconds_mean"]
        )["mean"]
        summary_row["peak_gpu_memory_mb_max"] = summarize_series(
            model_runs["peak_gpu_memory_mb_max"]
        )["max"]
        summary_rows.append(summary_row)

    return pd.DataFrame(summary_rows)


def write_empty_outputs() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    write_json(REPORTS_DIR / CASE_METRICS_FILENAME, [])
    write_json(REPORTS_DIR / MODEL_CASE_METRICS_FILENAME, [])
    write_json(REPORTS_DIR / RUN_SUMMARY_FILENAME, [])
    write_json(REPORTS_DIR / MODEL_SUMMARY_FILENAME, [])
    write_json(REPORTS_DIR / RUN_INVENTORY_FILENAME, [])


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    dataset_index = DatasetIndex()
    split_loader = SplitLoader()
    test_ids = split_loader.get_test_ids()
    run_specs = build_run_specs()

    case_frames = [
        evaluate_run(run_spec, test_ids, dataset_index)
        for run_spec in run_specs
    ]
    case_frames = [frame for frame in case_frames if not frame.empty]

    if not case_frames:
        print("CRITICAL: No prediction results found across any models.")
        write_empty_outputs()
        return

    case_metrics = pd.concat(case_frames, ignore_index=True)
    model_case_metrics = build_model_case_metrics(case_metrics)
    run_summary = build_run_summary(case_metrics)
    model_summary = build_model_summary(model_case_metrics, run_summary)
    run_inventory = pd.DataFrame([run_spec.inventory_record() for run_spec in run_specs])

    write_records_frame(REPORTS_DIR / CASE_METRICS_FILENAME, case_metrics)
    write_records_frame(REPORTS_DIR / MODEL_CASE_METRICS_FILENAME, model_case_metrics)
    write_records_frame(REPORTS_DIR / RUN_SUMMARY_FILENAME, run_summary)
    write_records_frame(REPORTS_DIR / MODEL_SUMMARY_FILENAME, model_summary)
    write_records_frame(REPORTS_DIR / RUN_INVENTORY_FILENAME, run_inventory)

    print("\n--- Aggregation Complete ---")
    print(f"Case-level rows written: {len(case_metrics)}")
    print(f"Run summaries written: {len(run_summary)}")
    print(f"Model summaries written: {len(model_summary)}")
    print(f"Outputs located in: {REPORTS_DIR}")


if __name__ == "__main__":
    main()
