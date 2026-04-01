from typing import Dict

import numpy as np
from medpy.metric import binary


def compute_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Dict[str, float]:
    """Compute overlap and boundary metrics for binary segmentation masks.

    The reporting contract for this benchmark requires overlap metrics together
    with a boundary metric pair. Empty-mask cases keep Dice/IoU semantics but
    surface-distance metrics remain undefined and are returned as `NaN`.

    Args:
        pred_mask: Predicted binary segmentation mask.
        gt_mask: Ground-truth binary segmentation mask.

    Returns:
        A dictionary with `dice`, `iou`, `hd95`, and `assd`.

    Raises:
        ValueError: If prediction and ground truth shapes differ.
    """
    if pred_mask.shape != gt_mask.shape:
        raise ValueError(
            f"Shape mismatch: prediction {pred_mask.shape} vs "
            f"ground truth {gt_mask.shape}. "
            f"Ensure prediction is resampled to reference before computing metrics."
        )

    pred = (pred_mask > 0).astype(np.bool_)
    gt = (gt_mask > 0).astype(np.bool_)

    pred_empty = not np.any(pred)
    gt_empty = not np.any(gt)

    metrics: Dict[str, float] = {
        "dice": 0.0,
        "iou": 0.0,
        "hd95": float("nan"),
        "assd": float("nan"),
    }

    if pred_empty and gt_empty:
        metrics["dice"] = 1.0
        metrics["iou"] = 1.0
        return metrics

    if pred_empty or gt_empty:
        return metrics

    metrics["dice"] = float(binary.dc(pred, gt))
    metrics["iou"] = float(binary.jc(pred, gt))
    metrics["hd95"] = float(binary.hd95(pred, gt))
    metrics["assd"] = float(binary.assd(pred, gt))
    return metrics
