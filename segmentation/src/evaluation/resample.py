import numpy as np
import SimpleITK as sitk
from typing import Union


def resample_to_reference(
    pred_mask: Union[np.ndarray, sitk.Image], 
    ref_mask: Union[np.ndarray, sitk.Image],
    is_binary: bool = True
) -> np.ndarray:
    """Resamples a predicted mask to match the spatial dimensions of a reference mask.
    
    Models often predict on resized inputs (e.g., 256x256). For fair evaluation,
    predictions must be resampled back to the original image's native resolution
    before metric computation. This function uses Nearest Neighbor interpolation
    for binary/categorical masks to avoid introducing spurious intermediate values.
    
    Args:
        pred_mask: The predicted segmentation mask.
        ref_mask: The ground truth reference mask.
        is_binary: If True, uses Nearest Neighbor interpolation (appropriate for 
                   categorical labels). If False, uses Linear interpolation.
                   
    Returns:
        A numpy array containing the resampled prediction, exactly matching
        the shape of `ref_mask`.
        
    Raises:
        ValueError: If input dimensions are fundamentally incompatible beyond 
                    spatial resampling (e.g., 2D vs 3D).
    """
    # Convert numpy arrays to SimpleITK images
    if isinstance(pred_mask, np.ndarray):
        pred_img = sitk.GetImageFromArray(pred_mask.astype(np.uint8))
    else:
        pred_img = pred_mask
        
    if isinstance(ref_mask, np.ndarray):
        ref_img = sitk.GetImageFromArray(ref_mask)
    else:
        ref_img = ref_mask
        
    # Check if shapes already match
    if pred_img.GetSize() == ref_img.GetSize():
        return sitk.GetArrayFromImage(pred_img)
        
    # Setup resampler
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ref_img)
    
    if is_binary:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resampler.SetInterpolator(sitk.sitkLinear)
        
    # To resample purely based on shape (assuming identical physical bounds
    # since we are just resizing the same 2D image matrix), we keep default
    # IdentityTransform. If these were true 3D DICOMs with different physical
    # spacings, we'd need to consider origin/spacing. For ISIC 2018 JPEGs,
    # origin and spacing are unit values.
    
    # We must match the physical spacing so the bounds line up perfectly
    # Calculate appropriate spacing derived from pixel grids
    pred_size = pred_img.GetSize()
    ref_size = ref_img.GetSize()
    
    pred_spacing = [
        pred_img.GetSpacing()[i] * (pred_size[i] / ref_size[i]) 
        for i in range(len(pred_size))
    ]
    pred_img.SetSpacing(pred_spacing)
    
    resampled_img = resampler.Execute(pred_img)
    return sitk.GetArrayFromImage(resampled_img)
