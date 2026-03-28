import yaml
import os
from typing import List, Dict, Optional, Any, Tuple
from monai.data import Dataset, CacheDataset
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Resized,
    NormalizeIntensityd,
    RandRotated,
    RandFlipd,
    RandZoomd,
    ToTensord,
    AsDiscreted
)

from src.dataio.dataset_index import DatasetIndex
from src.dataio.split_loader import SplitLoader

def get_unet_transforms(config_path: str, is_train: bool = True) -> Compose:
    """Creates a MONAI transformation pipeline based on the U-Net config.
    
    Sets up deterministic spatial transformations, intensity normalization,
    and target resizing. If `is_train` is True, appends random augmentations
    dictated by the configuration.
    
    Args:
        config_path: Path to the U-Net configuration YAML file.
        is_train: Whether to include training augmentations.
        
    Returns:
        A MONAI Compose object containing the transform pipeline.
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    prep_cfg = config.get('preprocessing', {})
    aug_cfg = config.get('augmentation', {})
    
    target_size = tuple(prep_cfg.get('target_size', [256, 256]))
    
    # Base transforms required for both train and validation
    transforms = [
        LoadImaged(keys=["image", "label"], image_only=True),
        EnsureChannelFirstd(keys=["image", "label"]),
        # Resize both image and label identically (nearest-neighbor for label)
        Resized(keys=["image", "label"], spatial_size=target_size, mode=("bilinear", "nearest")),
        # Optional: Normalize intensity using z-score or min-max depending on config
        NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=True)
    ]
    
    if is_train:
        # Augmentation transforms
        if 'rand_rotate' in aug_cfg:
            transforms.append(
                RandRotated(
                    keys=["image", "label"], 
                    prob=aug_cfg['rand_rotate'].get('prob', 0.5), 
                    range_x=aug_cfg['rand_rotate'].get('range_x', 0.261),
                    mode=("bilinear", "nearest"),
                    padding_mode="zeros"
                )
            )
        if 'rand_flip_h' in aug_cfg:
            transforms.append(
                RandFlipd(
                    keys=["image", "label"], 
                    spatial_axis=1, 
                    prob=aug_cfg['rand_flip_h'].get('prob', 0.5)
                )
            )
        if 'rand_flip_v' in aug_cfg:
            transforms.append(
                RandFlipd(
                    keys=["image", "label"], 
                    spatial_axis=0, 
                    prob=aug_cfg['rand_flip_v'].get('prob', 0.5)
                )
            )
        if 'rand_zoom' in aug_cfg:
            transforms.append(
                RandZoomd(
                    keys=["image", "label"], 
                    prob=aug_cfg['rand_zoom'].get('prob', 0.3),
                    min_zoom=aug_cfg['rand_zoom'].get('min_zoom', 0.9),
                    max_zoom=aug_cfg['rand_zoom'].get('max_zoom', 1.1),
                    mode=("bilinear", "nearest")
                )
            )
            
    # Finalize by strictly binarizing the label (in case interpolation caused artifacts)
    # Background=0, Lesion=1
    transforms.append(AsDiscreted(keys=["label"], threshold=0.5))
    transforms.append(ToTensord(keys=["image", "label"]))
    
    return Compose(transforms)

def create_datasets(
    unet_config_path: str = "configs/unet.yaml",
    dataset_config_path: str = "configs/dataset.yaml",
    split_path: str = "splits/primary_split.json",
    use_cache: bool = False
) -> Tuple[Dataset, Dataset, Dataset]:
    """Creates MONAI datasets for train, validation, and test splits.
    
    Reads the central split index, maps patient IDs to absolute paths,
    and applies the U-Net specific transformation pipelines.
    
    Args:
        unet_config_path: Path to the U-Net pipeline YAML.
        dataset_config_path: Path to the dataset context YAML.
        split_path: Path to the primary split JSON.
        use_cache: If True, uses CacheDataset instead of generic Dataset (faster but RAM heavy).
        
    Returns:
        A tuple of (train_dataset, val_dataset, test_dataset).
    """
    index = DatasetIndex(config_path=dataset_config_path)
    split_loader = SplitLoader(split_path=split_path)
    
    # Helper to build dict lists
    def _build_files_list(patient_ids: List[str]) -> List[Dict[str, str]]:
        result = []
        for pid in patient_ids:
            # We keep the patient_id in the dictionary for traceback/reporting
            paths = index.get_case(pid)
            result.append({
                "patient_id": pid,
                "image": paths["image_path"],
                "label": paths["mask_path"]
            })
        return result
        
    train_files = _build_files_list(split_loader.get_train_ids())
    val_files = _build_files_list(split_loader.get_val_ids())
    test_files = _build_files_list(split_loader.get_test_ids())
    
    train_transforms = get_unet_transforms(unet_config_path, is_train=True)
    eval_transforms = get_unet_transforms(unet_config_path, is_train=False)
    
    DatasetClass = CacheDataset if use_cache else Dataset
    
    train_ds = DatasetClass(data=train_files, transform=train_transforms)
    val_ds = DatasetClass(data=val_files, transform=eval_transforms)
    test_ds = DatasetClass(data=test_files, transform=eval_transforms)
    
    return train_ds, val_ds, test_ds
