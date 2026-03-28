import os
import yaml
from typing import Dict, TypedDict, Optional

class CasePaths(TypedDict):
    """Paths for a single medical image case."""
    image_path: str
    mask_path: str


class DatasetIndex:
    """Indexes and manages access to image and mask paths for the dataset.
    
    This class reads the dataset configuration to find the root paths and
    provides a unified interface to retrieve absolute paths for any patient ID.
    
    Attributes:
        config_path (str): Path to the dataset configuration YAML.
        config (Dict): Loaded configuration dictionary.
        root_path (str): Base path to the dataset.
        images_dir (str): Absolute path to the images directory.
        masks_dir (str): Absolute path to the ground truth masks directory.
        image_ext (str): File extension for images.
        mask_ext (str): File extension for masks.
    """
    
    def __init__(self, config_path: str = "configs/dataset.yaml") -> None:
        """Initialize the dataset index with a configuration file.
        
        Args:
            config_path: Relative or absolute path to the dataset.yaml config file.
            
        Raises:
            FileNotFoundError: If the config file cannot be found.
            KeyError: If required configuration keys are missing.
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)['dataset']
            
        self.root_path = self.config['root_path']
        self.images_dir = os.path.join(self.root_path, self.config['images_dir'])
        self.masks_dir = os.path.join(self.root_path, self.config['masks_dir'])
        self.image_ext = self.config['image_extension']
        self.mask_ext = self.config['mask_extension']
        
        if not os.path.exists(self.images_dir):
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        if not os.path.exists(self.masks_dir):
            raise FileNotFoundError(f"Masks directory not found: {self.masks_dir}")

    def get_case(self, patient_id: str) -> CasePaths:
        """Get the absolute paths to the image and mask for a specific patient.
        
        Args:
            patient_id: The unique identifier for the patient (e.g., 'ISIC_0000000').
            
        Returns:
            A dictionary containing the absolute paths to the image and mask.
            
        Raises:
            FileNotFoundError: If the image or mask file does not exist.
        """
        image_path = os.path.join(self.images_dir, f"{patient_id}{self.image_ext}")
        mask_path = os.path.join(self.masks_dir, f"{patient_id}{self.mask_ext}")
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found for patient {patient_id}: {image_path}")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask not found for patient {patient_id}: {mask_path}")
            
        return {
            "image_path": image_path,
            "mask_path": mask_path
        }
