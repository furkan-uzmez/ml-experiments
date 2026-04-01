from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CASE_METRICS_FILENAME = "case_metrics.json"
MODEL_CASE_METRICS_FILENAME = "model_case_metrics.json"
RUN_SUMMARY_FILENAME = "run_summary.json"
MODEL_SUMMARY_FILENAME = "model_summary.json"
RUN_INVENTORY_FILENAME = "run_inventory.json"
RUNTIME_LOG_FILENAME = "runtime.log"
TRAINING_HISTORY_FILENAME = "training_history.json"

SEGMENTATION_CLASS_NAME = "lesion"
PRIMARY_METRIC_NAME = "dice"
BOUNDARY_METRIC_NAMES = ("hd95", "assd")

RUN_METADATA_COLUMNS = (
    "run_id",
    "experiment_group",
    "model_name",
    "backbone",
    "task_type",
    "dataset_version",
    "split_version",
    "label_version",
    "seed",
    "loss_name",
    "optimizer",
    "learning_rate",
    "scheduler",
    "augmentation_summary",
    "preprocessing_summary",
    "threshold_policy",
    "threshold_used",
    "checkpoint_path",
    "train_started_at",
    "train_finished_at",
)

CASE_MEASUREMENT_COLUMNS = (
    "dice",
    "iou",
    "hd95",
    "assd",
    "lesion_count_gt",
    "lesion_count_pred",
    "volume_gt",
    "volume_pred",
    "inference_time_seconds",
    "peak_gpu_memory_mb",
)

CASE_PATH_COLUMNS = ("image_path", "mask_path", "prediction_path")


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    experiment_group: str
    model_name: str
    backbone: str
    task_type: str
    dataset_version: str
    split_version: str
    label_version: str
    seed: int
    loss_name: str
    optimizer: str
    learning_rate: float | None
    scheduler: str
    augmentation_summary: str
    preprocessing_summary: str
    threshold_policy: str
    threshold_used: float | None
    checkpoint_path: str
    prediction_dir: str
    prediction_suffix: str
    runtime_log_path: str | None
    train_started_at: str | None = None
    train_finished_at: str | None = None

    def inventory_record(self) -> dict[str, Any]:
        record = asdict(self)
        record.pop("prediction_dir", None)
        record.pop("prediction_suffix", None)
        record.pop("runtime_log_path", None)
        return record
