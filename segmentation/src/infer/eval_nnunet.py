import os
import sys
import argparse
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.dataio.dataset_index import DatasetIndex
from src.evaluation.metrics import compute_metrics
from src.evaluation.resample import resample_to_reference


def eval_nnunet_predictions(pred_dir: str, seed: int):
    """Evaluates nnU-Net's raw predictions against the original ISIC 2018 ground truth."""
    index = DatasetIndex()
    
    # nnU-Net saves predictions identically to the filename in imagesTs/imagesTr but without modality indicator
    # e.g., predictions/ISIC_0000000.png
    
    pred_files = [f for f in os.listdir(pred_dir) if f.endswith(".png")]
    print(f"Evaluating {len(pred_files)} predictions from {pred_dir} (Seed {seed})")
    
    results = {}
    dice_sum = 0.0
    iou_sum = 0.0
    hd95_sum = 0.0
    valid_hd95_count = 0
    
    for pf in tqdm(pred_files):
        # Filename format: ISIC_xxxxxxx.png
        patient_id = pf.replace(".png", "")
        pred_path = os.path.join(pred_dir, pf)
        
        # Ground Truth Phase 2 Index
        try:
            gt_path = index.get_case(patient_id)["mask_path"]
        except FileNotFoundError:
            print(f"Skipping {patient_id} - No GT available in source (Likely pure test dataset).")
            continue
            
        pred_img = sitk.ReadImage(pred_path)
        gt_img = sitk.ReadImage(gt_path)
        
        # Resample to native resolution. In nnUNet's case, if resampled incorrectly or at spacing diff
        # Phase 2 metrics logic enforces strict 2D NN matching.
        final_pred = resample_to_reference(pred_img, gt_img, is_binary=True)
        final_gt = sitk.GetArrayFromImage(gt_img)
        
        metrics = compute_metrics(final_pred, final_gt)
        
        results[patient_id] = metrics
        dice_sum += metrics['dice']
        iou_sum += metrics['iou']
        if not np.isnan(metrics['hd95']):
            hd95_sum += metrics['hd95']
            valid_hd95_count += 1
            
    num_eval = len(results)
    if num_eval == 0:
        print("No paired predictions evaluated.")
        return
        
    avg_dice = dice_sum / num_eval
    avg_iou = iou_sum / num_eval
    avg_hd95 = hd95_sum / valid_hd95_count if valid_hd95_count > 0 else float('nan')
    
    print(f"--- Final Evaluation (Seed {seed}) ---")
    print(f"Cases Evaluated: {num_eval}")
    print(f"Avg Dice:  {avg_dice:.4f}")
    print(f"Avg IoU:   {avg_iou:.4f}")
    print(f"Avg HD95:  {avg_hd95:.4f}")
    
    # Save results to a json
    out_json = os.path.join(pred_dir, f"metrics_seed_{seed}.json")
    with open(out_json, "w") as f:
        json.dump({
            "aggregate": {
                "avg_dice": avg_dice,
                "avg_iou": avg_iou,
                "avg_hd95": avg_hd95
            },
            "per_case": results
        }, f, indent=4)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Evaluate raw nnU-Net Output")
    parser.add_argument("--pred_dir", type=str, required=True, help="Path to nnU-Net predictions directory")
    parser.add_argument("--seed", type=int, required=True, help="Random seed for tracking")
    args = parser.parse_args()
    
    eval_nnunet_predictions(args.pred_dir, args.seed)
