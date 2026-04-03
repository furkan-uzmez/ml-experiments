import json
import os
import sys

import numpy as np
import yaml
from PIL import Image
import SimpleITK as sitk
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.models.medsam3_model import MedSAM3Model
from src.dataio.dataset_index import DatasetIndex
from src.dataio.split_loader import SplitLoader
from src.evaluation.metrics import compute_metrics
from src.evaluation.runtime import track_inference_time, track_peak_gpu_memory
from src.reporting.io_utils import write_jsonl_log
from src.reporting.reporting_contract import RUNTIME_LOG_FILENAME

@track_peak_gpu_memory
@track_inference_time
def tracked_predict(model, image_path):
    """Execute MedSAM3 inference while keeping runtime tracking isolated."""
    predicted_mask = model.predict(image_path)
    return {"mask": predicted_mask}

def infer_medsam():
    """Run the MedSAM3 benchmark inference loop over the test split.

    MedSAM3 is text-guided, so inference is driven by the configured concept
    prompts instead of ground-truth-derived bounding boxes.
    """
    with open("configs/medsam.yaml", "r") as f:
        config = yaml.safe_load(f)

    out_dir = config["inference"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    print("--- Starting MedSAM3 Benchmark Inference ---")

    model_config = dict(config["model"])
    model_config.update(config.get("prompting", {}))
    model = MedSAM3Model(config=model_config, device="cuda")

    index = DatasetIndex()
    split_loader = SplitLoader()
    test_ids = split_loader.get_test_ids()

    results = {}
    dice_sum, iou_sum, hd95_sum, assd_sum = 0.0, 0.0, 0.0, 0.0
    valid_hd95_count = 0
    valid_assd_count = 0
    inference_times = []
    peak_memories = []
    runtime_rows: list[dict] = []

    pbar = tqdm(test_ids, desc="Evaluating Test Split")
    for pid in pbar:
        case = index.get_case(pid)

        try:
            mask_img = sitk.ReadImage(case['mask_path'])
            mask_np = sitk.GetArrayFromImage(mask_img)
            gt_mask = (mask_np > 0).astype(np.uint8)
        except FileNotFoundError:
            print(f"Skipping {pid} - Missing Image/Label.")
            continue

        track_result = tracked_predict(model, case["image_path"])
        final_pred_np = track_result["mask"]
        inference_times.append(track_result["inference_time_seconds"])
        peak_memories.append(track_result["peak_gpu_memory_mb"])
        runtime_rows.append(
            {
                "run_id": "medsam3_zero_shot",
                "case_id": pid,
                "inference_time_seconds": track_result["inference_time_seconds"],
                "peak_gpu_memory_mb": track_result["peak_gpu_memory_mb"],
            }
        )

        metrics = compute_metrics(final_pred_np.astype(np.uint8), gt_mask)
        results[pid] = metrics
        dice_sum += metrics['dice']
        iou_sum += metrics['iou']
        if not np.isnan(metrics['hd95']):
            hd95_sum += metrics['hd95']
            valid_hd95_count += 1
        if not np.isnan(metrics['assd']):
            assd_sum += metrics['assd']
            valid_assd_count += 1
        out_mask_path = os.path.join(out_dir, f"{pid}_pred.png")
        pred_pil = Image.fromarray((final_pred_np * 255).astype(np.uint8), mode="L")
        pred_pil.save(out_mask_path)

    num_eval = len(results)
    if num_eval > 0:
        avg_dice = dice_sum / num_eval
        avg_iou = iou_sum / num_eval
        avg_hd95 = hd95_sum / valid_hd95_count if valid_hd95_count > 0 else float('nan')
        avg_assd = assd_sum / valid_assd_count if valid_assd_count > 0 else float('nan')
        avg_latency = float(np.mean(inference_times))
        max_mem = float(np.max(peak_memories))
        write_jsonl_log(os.path.join(out_dir, RUNTIME_LOG_FILENAME), runtime_rows)

        print("\n=== MedSAM3 Evaluation Complete ===")
        print(f"Avg Dice:  {avg_dice:.4f}")
        print(f"Avg IoU:   {avg_iou:.4f}")
        print(f"Avg HD95:  {avg_hd95:.4f}")
        print(f"Avg ASSD:  {avg_assd:.4f}")
        print(f"Avg Latency/Img: {avg_latency:.4f}s")
        print(f"Peak GPU Memory: {max_mem:.2f}MB")
        
        with open(os.path.join(out_dir, "benchmark_metrics.json"), "w") as f:
            json.dump({
                "aggregate": {
                    "avg_dice": avg_dice,
                    "avg_iou": avg_iou,
                    "avg_hd95": avg_hd95,
                    "avg_assd": avg_assd,
                    "avg_latency": avg_latency,
                    "peak_gpu_mb": max_mem
                },
                "per_case": results
            }, f, indent=4)
            

if __name__ == "__main__":
    infer_medsam()
