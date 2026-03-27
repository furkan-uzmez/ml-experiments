import numpy as np
from medpy.metric import binary
from typing import Dict, Any


def compute_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Dict[str, float]:
    """Computes standard medical image segmentation metrics on 2D/3D masks.
    
    Calculates Dice Similarity Coefficient (DSC), Intersection over Union (IoU),
    and 95th Percentile Hausdorff Distance (HD95).
    
    Handles metric edge cases robustly (e.g., empty masks) according to the
    benchmark contract in `configs/metrics.yaml`.
    
    Args:
        pred_mask: Predicted binary segmentation mask (numpy array).
                   Non-zero values are treated as the foreground class.
        gt_mask: Ground truth binary segmentation mask (numpy array).
                 Non-zero values are treated as the foreground class.
                 
    Returns:
        A dictionary containing the computed metrics:
            - 'dice': Dice Similarity Coefficient [0.0, 1.0]
            - 'iou': Jaccard Index [0.0, 1.0]
            - 'hd95': 95th Percentile Hausdorff Distance (pixels/voxels)
            
    Raises:
        ValueError: If the shapes of prediction and ground truth do not match.
    """
    if pred_mask.shape != gt_mask.shape:
        raise ValueError(
            f"Shape mismatch: prediction {pred_mask.shape} vs "
            f"ground truth {gt_mask.shape}. "
            f"Ensure prediction is resampled to reference before computing metrics."
        )
        
    # Binarize inputs strictly
    pred = (pred_mask > 0).astype(np.bool_)
    gt = (gt_mask > 0).astype(np.bool_)
    
    pred_empty = not np.any(pred)
    gt_empty = not np.any(gt)
    
    metrics: Dict[str, Any] = {
        'dice': 0.0,
        'iou': 0.0,
        'hd95': float('nan')
    }
    
    # --- Edge Case Handling ---
    if pred_empty and gt_empty:
        # Both empty: perfect match
        metrics['dice'] = 1.0
        metrics['iou'] = 1.0
        # HD95 remains NaN as distance between empty sets is undefined
        return metrics
        
    if pred_empty and not gt_empty:
        # Missed entirely
        return metrics
        
    if not pred_empty and gt_empty:
        # Hallucination
        return metrics
        
    # --- Standard Computation ---
    # Both sets have foreground pixels; medpy handles this safely.
    metrics['dice'] = float(binary.dc(pred, gt))
    metrics['iou'] = float(binary.jc(pred, gt))
    
    # Calculate HD95. Note: Medpy calculates distance in physical space if
    # voxelspacing is provided. Default is 1.0 pixel spacing.
    metrics['hd95'] = float(binary.hd95(pred, gt))
    
    return metrics
