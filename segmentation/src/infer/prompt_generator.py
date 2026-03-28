import numpy as np
from typing import Optional, Tuple

class PromptGenerator:
    """Generates deterministic bounding box prompts from ground truth masks.
    
    In a zero-shot evaluation, foundation models like MedSAM require tight 
    spatial conditioning. This class mimics a typical human bounding box 
    annotation by calculating the extreme extents of the ground truth mask 
    and applying a standardized padding on all sides.
    """
    
    def __init__(self, padding: int = 5, strategy: str = "macro"):
        """Initializes the prompt generation logic.
        
        Args:
            padding: Pixels to add uniformly to the absolute max/min bounding coordinates.
            strategy: 'macro' (wrap all positive pixels) or component-wise (unimplemented).
        """
        self.padding = padding
        self.strategy = strategy
        
    def generate_bbox(self, gt_mask: np.ndarray) -> Optional[np.ndarray]:
        """Calculates a loose bounding box encompassing the 2D mask.
        
        Args:
            gt_mask: A 2D numpy array where >0 indicates the lesion.
            
        Returns:
            A 1D numpy array `[x_min, y_min, x_max, y_max]` representing the
            corners of the prompt box. Returns None if the mask is entirely empty.
        """
        if self.strategy != "macro":
            raise NotImplementedError("Only 'macro' boxing strategy is currently supported for ISIC 2018.")
            
        # Ensure boolean and compute coordinates of all positive pixels
        ys, xs = np.where(gt_mask > 0)
        
        # Guard against fully empty masks
        if len(xs) == 0 or len(ys) == 0:
            return None
            
        y_min, y_max = np.min(ys), np.max(ys)
        x_min, x_max = np.min(xs), np.max(xs)
        
        # Apply padding
        x_min = max(0, x_min - self.padding)
        y_min = max(0, y_min - self.padding)
        
        # Guard against exceeding right/bottom resolution
        H, W = gt_mask.shape
        x_max = min(W - 1, x_max + self.padding)
        y_max = min(H - 1, y_max + self.padding)
        
        return np.array([x_min, y_min, x_max, y_max])
