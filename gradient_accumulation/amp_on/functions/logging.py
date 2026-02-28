"""Structured experiment logging utilities for training comparisons."""

from __future__ import annotations

import csv
import json
import logging as pylogging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch

try:
    import psutil
except Exception:  # pragma: no cover - optional dependency
    psutil = None


STEP_FIELDS = [
    "timestamp_utc",
    "run_name",
    "phase",
    "epoch",
    "global_step",
    "optimizer_step",
    "loss",
    "accuracy",
    "step_time_sec",
    "samples",
    "batches",
    "samples_per_sec",
    "batches_per_sec",
    "lr",
    "accumulation_steps",
]

EPOCH_FIELDS = [
    "timestamp_utc",
    "run_name",
    "phase",
    "epoch",
    "loss",
    "accuracy",
    "epoch_time_sec",
    "avg_step_time_sec",
    "samples",
    "batches",
    "samples_per_sec",
    "batches_per_sec",
    "optimizer_steps",
    "peak_vram_mb",
]

SYSTEM_FIELDS = [
    "timestamp_utc",
    "run_name",
    "phase",
    "epoch",
    "global_step",
    "cpu_percent",
    "ram_percent",
    "ram_used_mb",
    "process_rss_mb",
    "gpu_allocated_mb",
    "gpu_reserved_mb",
    "gpu_max_allocated_mb",
]


@dataclass(frozen=True)
class ExperimentLoggerConfig:
    """Configuration for structured run logging.

    Attributes:
        run_name: Human readable experiment identifier.
        output_dir: Root directory where logs will be written.
        overwrite: Whether to overwrite existing run directory files.
    """

    run_name: str
    output_dir: str | os.PathLike[str] = "runs"
    overwrite: bool = False


def now_utc_iso() -> str:
    """Return current UTC timestamp in ISO-8601 format."""

    return datetime.now(timezone.utc).isoformat()


def _coerce_row(row: Mapping[str, Any], fieldnames: list[str]) -> Dict[str, Any]:
    """Normalize row values to a fixed CSV schema."""

    return {name: row.get(name) for name in fieldnames}


def _append_csv(path: Path, row: Mapping[str, Any], fieldnames: list[str]) -> None:
    """Append one row to a CSV file and add header if file is new."""

    path.parent.mkdir(parents=True, exist_ok=True)
    row_data = _coerce_row(row, fieldnames)
    file_exists = path.exists()

    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_data)


def get_text_logger(log_path: str | os.PathLike[str]) -> pylogging.Logger:
    """Create a text logger that writes to both console and file.

    Args:
        log_path: Output log file path.

    Returns:
        Configured `logging.Logger` instance.
    """

    logger_name = f"training_logger::{Path(log_path).resolve()}"
    logger = pylogging.getLogger(logger_name)
    logger.setLevel(pylogging.INFO)
    logger.propagate = False

    if logger.handlers:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()

    formatter = pylogging.Formatter(
        "[%(asctime)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = pylogging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)

    console_handler = pylogging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def get_logger(log_path: str | os.PathLike[str] = "training.log") -> pylogging.Logger:
    """Backward-compatible alias for file+console text logger."""

    return get_text_logger(log_path=log_path)


def collect_system_metrics(
    device: Optional[torch.device] = None,
    run_name: Optional[str] = None,
    phase: Optional[str] = None,
    epoch: Optional[int] = None,
    global_step: Optional[int] = None,
) -> Dict[str, Any]:
    """Collect CPU/RAM/GPU usage metrics for telemetry logging.

    Args:
        device: Training device, used for CUDA memory stats.
        run_name: Optional experiment identifier.
        phase: Optional phase label (`train`, `val`, etc.).
        epoch: Optional epoch index.
        global_step: Optional global step index.

    Returns:
        Dictionary containing host and GPU utilization metrics.
    """

    cpu_percent = None
    ram_percent = None
    ram_used_mb = None
    process_rss_mb = None

    if psutil is not None:
        cpu_percent = psutil.cpu_percent(interval=None)
        vm = psutil.virtual_memory()
        ram_percent = vm.percent
        ram_used_mb = vm.used / (1024.0 * 1024.0)
        process = psutil.Process(os.getpid())
        process_rss_mb = process.memory_info().rss / (1024.0 * 1024.0)

    gpu_allocated_mb = None
    gpu_reserved_mb = None
    gpu_max_allocated_mb = None

    if device is not None and device.type == "cuda" and torch.cuda.is_available():
        index = device.index if device.index is not None else torch.cuda.current_device()
        gpu_allocated_mb = torch.cuda.memory_allocated(index) / (1024.0 * 1024.0)
        gpu_reserved_mb = torch.cuda.memory_reserved(index) / (1024.0 * 1024.0)
        gpu_max_allocated_mb = torch.cuda.max_memory_allocated(index) / (1024.0 * 1024.0)

    return {
        "timestamp_utc": now_utc_iso(),
        "run_name": run_name,
        "phase": phase,
        "epoch": epoch,
        "global_step": global_step,
        "cpu_percent": cpu_percent,
        "ram_percent": ram_percent,
        "ram_used_mb": ram_used_mb,
        "process_rss_mb": process_rss_mb,
        "gpu_allocated_mb": gpu_allocated_mb,
        "gpu_reserved_mb": gpu_reserved_mb,
        "gpu_max_allocated_mb": gpu_max_allocated_mb,
    }


class ExperimentLogger:
    """Logger that persists experiment telemetry in text, CSV and JSON formats."""

    def __init__(self, config: ExperimentLoggerConfig) -> None:
        """Initialize output paths and text logger.

        Args:
            config: Logging configuration.
        """

        self.config = config
        self.run_name = config.run_name
        self.run_dir = Path(config.output_dir) / config.run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.step_csv = self.run_dir / "step_metrics.csv"
        self.epoch_csv = self.run_dir / "epoch_metrics.csv"
        self.system_csv = self.run_dir / "system_metrics.csv"
        self.summary_json = self.run_dir / "run_summary.json"
        self.text_log_path = self.run_dir / "run.log"

        if config.overwrite:
            for path in (
                self.step_csv,
                self.epoch_csv,
                self.system_csv,
                self.summary_json,
                self.text_log_path,
            ):
                if path.exists():
                    path.unlink()

        self.logger = get_text_logger(self.text_log_path)

    def info(self, message: str) -> None:
        """Write a message to console and text log."""

        self.logger.info(message)

    def log_step(self, row: Mapping[str, Any]) -> None:
        """Append a training/validation step row to `step_metrics.csv`."""

        payload = {"timestamp_utc": now_utc_iso(), "run_name": self.run_name, **row}
        _append_csv(self.step_csv, payload, STEP_FIELDS)

    def log_epoch(self, row: Mapping[str, Any]) -> None:
        """Append an epoch summary row to `epoch_metrics.csv`."""

        payload = {"timestamp_utc": now_utc_iso(), "run_name": self.run_name, **row}
        _append_csv(self.epoch_csv, payload, EPOCH_FIELDS)

    def log_system(self, row: Mapping[str, Any]) -> None:
        """Append a system telemetry row to `system_metrics.csv`."""

        payload = {"timestamp_utc": now_utc_iso(), "run_name": self.run_name, **row}
        _append_csv(self.system_csv, payload, SYSTEM_FIELDS)

    def log_system_snapshot(
        self,
        device: Optional[torch.device],
        phase: str,
        epoch: Optional[int],
        global_step: Optional[int],
    ) -> Dict[str, Any]:
        """Collect and persist one system metrics snapshot.

        Returns:
            The same metrics dictionary that is written to disk.
        """

        payload = collect_system_metrics(
            device=device,
            run_name=self.run_name,
            phase=phase,
            epoch=epoch,
            global_step=global_step,
        )
        self.log_system(payload)
        return payload

    def write_summary(self, summary: Mapping[str, Any]) -> None:
        """Persist final experiment summary to JSON."""

        self.summary_json.parent.mkdir(parents=True, exist_ok=True)
        with self.summary_json.open("w", encoding="utf-8") as handle:
            json.dump(dict(summary), handle, indent=2, ensure_ascii=True)

