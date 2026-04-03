from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.dataio.split_loader import SplitLoader
from src.reporting.io_utils import load_records_frame
from src.reporting.reporting_contract import (
    CASE_METRICS_FILENAME,
    MODEL_CASE_METRICS_FILENAME,
    MODEL_SUMMARY_FILENAME,
    RUN_INVENTORY_FILENAME,
    RUN_SUMMARY_FILENAME,
)

REPORTS_DIR = Path("reports")
FIGURES_DIR = REPORTS_DIR / "figures"
PRIMARY_METRIC = "Dice"


def read_dataset_config() -> dict:
    with open("configs/dataset.yaml", "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)["dataset"]


def format_float(value: float, digits: int = 4) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value:.{digits}f}"


def sort_for_ranking(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.assign(
        hd95_rank=frame["hd95_mean"].fillna(np.inf),
        assd_rank=frame["assd_mean"].fillna(np.inf),
    ).sort_values(
        by=["dice_mean", "hd95_rank", "assd_rank"],
        ascending=[False, True, True],
    )


def choose_recommendation(
    model_summary: pd.DataFrame,
    run_summary: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    ranked_models = sort_for_ranking(model_summary)
    recommended_model = ranked_models.iloc[0]
    recommended_run = sort_for_ranking(
        run_summary[run_summary["model_name"] == recommended_model["model_name"]]
    ).iloc[0]
    return recommended_model, recommended_run


def build_task_fingerprint(dataset_config: dict, run_inventory: pd.DataFrame) -> str:
    split_loader = SplitLoader(dataset_config["split_file"])
    train_count = len(split_loader.get_train_ids())
    val_count = len(split_loader.get_val_ids())
    test_count = len(split_loader.get_test_ids())
    threshold_rules = sorted(
        run_inventory["threshold_policy"].dropna().astype(str).unique().tolist()
    )
    label_version = run_inventory["label_version"].dropna().iloc[0]
    split_version = run_inventory["split_version"].dropna().iloc[0]

    lines = [
        "## 1. Task Fingerprint",
        "",
        f"- task type: {dataset_config.get('task_type', 'binary_segmentation')}",
        "- clinical objective: lesion boundary segmentation with contour-aware evaluation",
        "- evaluation unit: image-level case (ISIC image corresponds to one patient)",
        f"- dataset version: {dataset_config.get('name', 'ISIC2018_Task1')}",
        f"- split version: {split_version} ({train_count} train / {val_count} val / {test_count} test)",
        f"- label version: {label_version} (inferred from dataset config because explicit label versioning is not recorded yet)",
        f"- primary metric: {PRIMARY_METRIC}",
        f"- threshold policy: {'; '.join(threshold_rules)}",
        "- note: probability-based calibration analysis is unavailable because the current benchmark exports hard masks only",
    ]
    return "\n".join(lines)


def build_runs_compared_table(run_summary: pd.DataFrame) -> str:
    header = [
        "## 2. Runs Compared",
        "",
        "| Run | Model | Seed | Loss | Main threshold rule | Notes |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    rows = []
    for _, row in run_summary.sort_values(["model_name", "seed"]).iterrows():
        if "MedSAM" in row["model_name"]:
            notes = "Text-conditioned inference with medical concept prompts; zero-shot unless configured otherwise."
        elif row["model_name"] == "nnU-Net":
            notes = "Self-configuring pipeline; runtime not captured in this integration."
        else:
            notes = "Supervised baseline with repeated seeds."

        rows.append(
            "| {run_id} | {model_name} | {seed} | {loss_name} | {threshold_policy} | {notes} |".format(
                run_id=row["run_id"],
                model_name=row["model_name"],
                seed=int(row["seed"]),
                loss_name=row["loss_name"],
                threshold_policy=row["threshold_policy"],
                notes=notes,
            )
        )

    return "\n".join(header + rows)


def build_core_metrics_table(model_summary: pd.DataFrame) -> str:
    header = [
        "## 3. Core Metrics",
        "",
        "| Model | Split | Primary metric | Secondary metrics | Calibration | Efficiency notes |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    rows = []
    for _, row in sort_for_ranking(model_summary).iterrows():
        primary = f"Dice {format_float(row['dice_mean'])} ± {format_float(row['dice_std'])}"
        secondary = (
            f"IoU {format_float(row['iou_mean'])} ± {format_float(row['iou_std'])}; "
            f"HD95 {format_float(row['hd95_mean'], 2)}; "
            f"ASSD {format_float(row['assd_mean'], 2)}"
        )
        calibration = "Unavailable; no probabilities/logits persisted for final evaluation"
        efficiency = (
            f"Latency {format_float(row['inference_time_seconds_mean'], 3)} s/case; "
            f"Peak GPU {format_float(row['peak_gpu_memory_mb_max'], 1)} MB"
        )
        rows.append(
            f"| {row['model_name']} | test | {primary} | {secondary} | {calibration} | {efficiency} |"
        )
    return "\n".join(header + rows)


def build_threshold_section(model_case_metrics: pd.DataFrame, run_inventory: pd.DataFrame) -> str:
    if model_case_metrics.empty or run_inventory.empty:
        return "\n".join(
            [
                "## 4. Threshold And Calibration Analysis",
                "",
                "- selected operating point: unavailable",
                "- justification: unavailable because reporting inventory is empty.",
                f"- missed-case risk: boundary misses should be read alongside {PRIMARY_METRIC}.",
                "- false-positive burden: unavailable",
                "- calibration summary: not available in the current pipeline because logits/probabilities are not persisted at case level.",
            ]
        )

    threshold_policies = [
        f"{row.model_name}: {row.threshold_policy}"
        for row in run_inventory.drop_duplicates(subset=["model_name"])[["model_name", "threshold_policy"]].itertuples()
    ]
    volume_bias = (
        model_case_metrics.assign(volume_bias=model_case_metrics["volume_pred"] - model_case_metrics["volume_gt"])
        .groupby("model_name")["volume_bias"]
        .mean()
        .sort_values()
    )
    volume_bias_lines = [
        f"{model}: mean volume bias {value:+.1f} foreground pixels"
        for model, value in volume_bias.items()
    ]

    return "\n".join(
        [
            "## 4. Threshold And Calibration Analysis",
            "",
            f"- selected operating point: {'; '.join(threshold_policies)}",
            "- justification: the benchmark compares published default decision rules instead of post-hoc threshold tuning to keep runs like-for-like.",
            f"- missed-case risk: boundary misses should be read alongside {PRIMARY_METRIC} because hard-mask outputs prevent threshold sweeps.",
            f"- false-positive burden: {'; '.join(volume_bias_lines)}",
            "- calibration summary: not available in the current pipeline because logits/probabilities are not persisted at case level.",
        ]
    )


def add_figure_section(
    sections: list[str],
    title: str,
    filename: str,
    interpretation: str,
) -> None:
    figure_path = FIGURES_DIR / filename
    if not figure_path.exists():
        return

    sections.extend(
        [
            f"### {title}",
            interpretation,
            f"![{title}](figures/{filename})",
            "",
        ]
    )


def build_small_lesion_note(model_case_metrics: pd.DataFrame, model_name: str) -> str:
    frame = model_case_metrics[model_case_metrics["model_name"] == model_name]
    if frame.empty:
        return "Small-lesion analysis unavailable."

    median_volume = frame["volume_gt"].median()
    small_cases = frame[frame["volume_gt"] <= median_volume]
    large_cases = frame[frame["volume_gt"] > median_volume]
    if small_cases.empty or large_cases.empty:
        return "Small-lesion analysis unavailable."

    return (
        "Cases at or below the median lesion size reach Dice "
        f"{format_float(small_cases['dice'].mean())}, compared with "
        f"{format_float(large_cases['dice'].mean())} for larger lesions."
    )


def build_figure_gallery(
    model_summary: pd.DataFrame,
    model_case_metrics: pd.DataFrame,
    run_summary: pd.DataFrame,
) -> str:
    recommended_model, _ = choose_recommendation(model_summary, run_summary)
    sections = [
        "## 5. Figures",
        "",
    ]
    add_figure_section(
        sections,
        "Learning Curves",
        "learning_curves.png",
        "U-Net learning curves are included when epoch-level history is available; nnU-Net and zero-shot MedSAM3 do not expose comparable training traces in this codebase.",
    )
    add_figure_section(
        sections,
        "Dice Distribution",
        "dice_distribution.png",
        f"{recommended_model['model_name']} leads on case-level overlap after averaging repeated seeds by case.",
    )
    add_figure_section(
        sections,
        "Boundary Metrics",
        "boundary_distributions.png",
        "HD95 and ASSD are shown next to Dice so contour accuracy is not hidden by overlap-only reporting.",
    )
    add_figure_section(
        sections,
        "Lesion Size vs Performance",
        "lesion_size_vs_dice.png",
        build_small_lesion_note(model_case_metrics, recommended_model["model_name"]),
    )
    add_figure_section(
        sections,
        "Overlay Gallery",
        "overlay_gallery.png",
        "Best, median, and worst cases are drawn from the strongest run of each model for qualitative inspection.",
    )
    return "\n".join(sections)


def build_failure_cases(
    case_metrics: pd.DataFrame,
    model_case_metrics: pd.DataFrame,
    model_summary: pd.DataFrame,
    run_summary: pd.DataFrame,
) -> str:
    if case_metrics.empty or model_case_metrics.empty:
        return "\n".join(
            [
                "## 6. Failure Cases",
                "",
                "- common false positives: unavailable",
                "- common false negatives: unavailable",
                "- subgroup weaknesses: unavailable",
                "- label quality concerns: unavailable because case-level reporting inputs are empty.",
            ]
        )

    recommended_model, recommended_run = choose_recommendation(model_summary, run_summary)
    worst_cases = (
        case_metrics[case_metrics["run_id"] == recommended_run["run_id"]]
        .sort_values("dice", ascending=True)
        .head(3)[["case_id", "dice", "hd95", "assd"]]
    )
    worst_case_lines = [
        f"{row.case_id} (Dice {row.dice:.3f}, HD95 {row.hd95:.2f}, ASSD {row.assd:.2f})"
        for row in worst_cases.itertuples()
    ]

    volume_bias = model_case_metrics.assign(
        volume_bias=model_case_metrics["volume_pred"] - model_case_metrics["volume_gt"]
    ).groupby("model_name")["volume_bias"].mean()
    over_segmenter = volume_bias.sort_values(ascending=False).index[0]
    under_segmenter = volume_bias.sort_values(ascending=True).index[0]

    return "\n".join(
        [
            "## 6. Failure Cases",
            "",
            f"- common false positives: {over_segmenter} shows the largest positive mean volume bias ({volume_bias[over_segmenter]:+.1f} foreground pixels).",
            f"- common false negatives: {under_segmenter} shows the largest negative mean volume bias ({volume_bias[under_segmenter]:+.1f} foreground pixels).",
            f"- subgroup weaknesses: {build_small_lesion_note(model_case_metrics, recommended_model['model_name'])}",
            f"- label quality concerns: no automatic label-quality audit is implemented; inspect difficult cases from {recommended_run['run_id']} first: {'; '.join(worst_case_lines)}.",
        ]
    )


def build_recommendation(model_summary: pd.DataFrame, run_summary: pd.DataFrame) -> str:
    recommended_model, recommended_run = choose_recommendation(model_summary, run_summary)
    tradeoff = (
        "probability-aware thresholding and calibration remain unavailable because the pipeline exports hard masks only"
    )
    next_step = (
        "persist logits/probability maps per case and add lesion-wise failure analysis for sparse or tiny lesions"
    )

    return "\n".join(
        [
            "## 7. Recommendation",
            "",
            f"- best run for the stated objective: {recommended_model['model_name']} (representative run `{recommended_run['run_id']}`)",
            f"- why it wins: highest mean Dice ({format_float(recommended_model['dice_mean'])}) with supporting boundary metrics HD95 {format_float(recommended_model['hd95_mean'], 2)} and ASSD {format_float(recommended_model['assd_mean'], 2)}.",
            f"- what tradeoff remains: {tradeoff}",
            f"- what to test next: {next_step}",
        ]
    )


def generate_markdown_report(out_dir: str) -> None:
    case_metrics = load_records_frame(REPORTS_DIR / CASE_METRICS_FILENAME)
    model_case_metrics = load_records_frame(REPORTS_DIR / MODEL_CASE_METRICS_FILENAME)
    run_summary = load_records_frame(REPORTS_DIR / RUN_SUMMARY_FILENAME)
    model_summary = load_records_frame(REPORTS_DIR / MODEL_SUMMARY_FILENAME)
    run_inventory = load_records_frame(REPORTS_DIR / RUN_INVENTORY_FILENAME)

    if (
        case_metrics.empty
        or model_case_metrics.empty
        or model_summary.empty
        or run_summary.empty
        or run_inventory.empty
    ):
        print("Missing reporting inputs. Skipping markdown report generation.")
        return

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    dataset_config = read_dataset_config()

    sections = [
        f"# {dataset_config.get('name', 'ISIC2018_Task1')} - Medical DL Experiment Report",
        "",
        f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        build_task_fingerprint(dataset_config, run_inventory),
        "",
        build_runs_compared_table(run_summary),
        "",
        build_core_metrics_table(model_summary),
        "",
        build_threshold_section(model_case_metrics, run_inventory),
        "",
        build_figure_gallery(model_summary, model_case_metrics, run_summary),
        "",
        build_failure_cases(case_metrics, model_case_metrics, model_summary, run_summary),
        "",
        build_recommendation(model_summary, run_summary),
        "",
    ]

    output_path = Path(out_dir) / "BENCHMARK_REPORT.md"
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(sections))

    print(f"Report generated successfully -> {output_path}")


if __name__ == "__main__":
    generate_markdown_report("reports")
