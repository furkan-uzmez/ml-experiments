from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def load_json(path: str | Path, default: Any) -> Any:
    file_path = Path(path)
    if not file_path.exists() or file_path.stat().st_size == 0:
        return default

    with open(file_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, payload: Any) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_records_frame(path: str | Path) -> pd.DataFrame:
    payload = load_json(path, [])
    if isinstance(payload, list):
        return pd.DataFrame(payload)
    return pd.DataFrame()


def write_records_frame(path: str | Path, frame: pd.DataFrame) -> None:
    records = frame.to_dict(orient="records") if not frame.empty else []
    write_json(path, records)


def write_jsonl_log(path: str | Path, records: list[dict[str, Any]]) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def load_jsonl_log(path: str | Path) -> list[dict[str, Any]]:
    file_path = Path(path)
    if not file_path.exists() or file_path.stat().st_size == 0:
        return []

    records: list[dict[str, Any]] = []
    with open(file_path, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            records.append(json.loads(stripped))
    return records
