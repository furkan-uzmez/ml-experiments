import os
import sys
import pandas as pd
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.dataio.dataset_index import DatasetIndex
from src.dataio.split_loader import SplitLoader
from src.evaluation.metrics import compute_metrics
from src.evaluation.resample import resample_to_reference


def evaluate_predictions_dir(model_name: str, seed: int, pred_dir: str, suffix: str, test_ids: list, index: DatasetIndex) -> pd.DataFrame:
    """Evaluates a directory of predictions against Ground Truth."""
    if not os.path.exists(pred_dir):
        print(f"Warning: Directory not found - {pred_dir}. Skipping...")
        return pd.DataFrame()
        
    records = []
    
    print(f"Aggregating {model_name} (Seed {seed}) from {pred_dir}...")
    for pid in tqdm(test_ids):
        pred_path = os.path.join(pred_dir, f"{pid}{suffix}")
        
        if not os.path.exists(pred_path):
            continue
            
        gt_path = index.get_case(pid)["mask_path"]
        if not os.path.exists(gt_path):
            continue
            
        pred_img = sitk.ReadImage(pred_path)
        gt_img = sitk.ReadImage(gt_path)
        
        # Resample logic standard enforcement
        try:
            final_pred = resample_to_reference(pred_img, gt_img, is_binary=True)
            final_gt = sitk.GetArrayFromImage(gt_img)
            
            final_pred = (final_pred > 0).astype(np.uint8)
            final_gt = (final_gt > 0).astype(np.uint8)
            
            met = compute_metrics(final_pred, final_gt)
            
            records.append({
                "model": model_name,
                "seed": seed,
                "case_id": pid,
                "dice": met['dice'],
                "iou": met['iou'],
                "hd95": met['hd95']
            })
        except Exception as e:
            print(f"Failed on {pid}: {e}")
            
    return pd.DataFrame(records)


def main():
    """Aggregates all evaluation predictions from U-Net, nnU-Net, and MedSAM."""
    out_dir = "reports"
    os.makedirs(out_dir, exist_ok=True)
    
    index = DatasetIndex()
    loader = SplitLoader()
    test_ids = loader.get_test_ids()
    
    seeds = [11, 22, 33]
    all_dfs = []
    
    # 1. U-Net Aggregation
    for s in seeds:
        pred_dir = f"artifacts/unet/predictions/seed_{s}"
        # infer_unet.py saves as ISIC_0000000_segmentation.png
        df = evaluate_predictions_dir("U-Net", s, pred_dir, "_segmentation.png", test_ids, index)
        if not df.empty:
            all_dfs.append(df)
            
    # 2. nnU-Net Aggregation
    for s in seeds:
        pred_dir = f"artifacts/nnunet/nnUNet_results/Dataset500_ISIC2018/predictions_seed_{s}"
        # eval_nnunet.py assumes format ISIC_0000000.png
        df = evaluate_predictions_dir("nnU-Net", s, pred_dir, ".png", test_ids, index)
        if not df.empty:
            all_dfs.append(df)
            
    # 3. MedSAM Aggregation (Zero-Shot)
    medsam_dir = "artifacts/medsam/predictions"
    # infer_medsam.py saves as ISIC_0000000_pred.png
    df = evaluate_predictions_dir("MedSAM", 0, medsam_dir, "_pred.png", test_ids, index)
    if not df.empty:
        all_dfs.append(df)
        
    if not all_dfs:
        print("CRITICAL: No prediction results found across any models.")
        # We will write an empty CSV to avoid pipeline crashes
        pd.DataFrame(columns=["model", "seed", "case_id", "dice", "iou", "hd95"]).to_csv(os.path.join(out_dir, "benchmark_results.csv"), index=False)
        return
        
    final_df = pd.concat(all_dfs, ignore_index=True)
    out_path = os.path.join(out_dir, "benchmark_results.csv")
    final_df.to_csv(out_path, index=False)
    
    print(f"\n--- Aggregation Complete ---")
    print(f"Total evaluated inferences: {len(final_df)}")
    print(f"Outputs written to: {out_path}")


if __name__ == "__main__":
    main()
