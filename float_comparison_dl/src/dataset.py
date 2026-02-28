import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from src.utils import setup_logger
import os

log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
logger = setup_logger(__name__, log_file=os.path.join(log_dir, "dataset.log"))

class COVIDCXNetDataset(Dataset):
    """
    COVID-CXNet Dataset for binary classification (AP vs PA).
    """
    
    # Binary class mapping: AP=0, PA=1
    CLASSES = ['AP', 'PA']
    CLASS_TO_IDX = {'AP': 0, 'PA': 1}
    
    def __init__(self, csv_file, root_dir, transform=None, split='all'):
        """
        Args:
            csv_file: Path to covidx_merged.csv
            root_dir: Root directory containing the images (usually the parent of covidx folder if filepath starts with covidx/)
                      or the covidx folder itself if filepath names are relative to it.
                      Based on CSV check, filepaths are like 'covidx/train/...', so root_dir should be the parent of 'covidx'.
            transform: Optional transforms to apply
            split: One of 'train', 'val', 'test', 'all'
        """
        self.root_dir = root_dir
        self.transform = transform
        
        # Load CSV
        try:
            self.data = pd.read_csv(csv_file)
        except Exception as e:
            raise IOError(f"Error loading CSV file: {csv_file}. {e}")

        # Filter by split
        if split and split.lower() != 'all':
            if 'split' in self.data.columns:
                self.data = self.data[self.data['split'] == split].reset_index(drop=True)
            else:
                logger.warning(f"Warning: 'split' column not found in CSV. Using full dataset.")

        # Filter for labels AP and PA in 'projection' column
        if 'projection' in self.data.columns:
            valid_labels = self.data['projection'].isin(['PA', 'AP'])
            logger.info(f"Original samples: {len(self.data)}")
            logger.info(f"Valid AP/PA samples: {valid_labels.sum()}")

            
            if not valid_labels.all():
                self.data = self.data[valid_labels].reset_index(drop=True)
        else:
             raise ValueError("CSV does not contain 'projection' column required for labels.")
        
        # Class info
        self.classes = self.CLASSES
        self.class_to_idx = self.CLASS_TO_IDX
        
        logger.info(f"Loaded {len(self.data)} samples for split '{split}'")
        logger.info(f"Class distribution: PA={len(self.data[self.data['projection'] == 'PA'])}, "
              f"AP={len(self.data[self.data['projection'] == 'AP'])}")

        # Verify image existence logic
        # Assuming filepath column is like 'covidx/train/filename.jpg'
        # And root_dir is supposed to be the directory containing 'covidx'
        if 'filepath' not in self.data.columns:
             raise ValueError("CSV does not contain 'filepath' column")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Construct image path
        # filepath is usually 'covidx/train/filename'
        rel_path = row['filepath']
        img_path = os.path.join(self.root_dir, rel_path)
        
        # Label
        label_str = row['projection']
        label = self.class_to_idx[label_str]
        
        try:
            image = Image.open(img_path)
            
            # Ensure RGB
            image = image.convert("RGB")
            
            # Apply transforms
            if self.transform:
                image = self.transform(image)
                
            return image, label
            
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            # Identify a failed sample - in a real training loop, you might want to return None or skip
            # For now, return a dummy tensor or raise
            raise e

def get_dataloaders(csv_file, root_dir, batch_size=32, num_workers=4, transform=None):
    """
    Utility function to create dataloaders for the COVIDCXNetDataset.
    """
    train_ds = COVIDCXNetDataset(csv_file, root_dir, transform=transform, split='train')
    val_ds = COVIDCXNetDataset(csv_file, root_dir, transform=transform, split='val')
    test_ds = COVIDCXNetDataset(csv_file, root_dir, transform=transform, split='test')
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    # Test block
    # Note: Update these paths to manual local paths if needed
    root = "/home/furkan/Projects/NIHChestXrays/COVID-CXNet"
    csv_path = os.path.join(root, "covidx/covidx_merged.csv")
    
    if os.path.exists(csv_path):
        print(f"Testing COVIDCXNetDataset with {csv_path}...")
        try:
            ds = COVIDCXNetDataset(csv_file=csv_path, root_dir=root, split='train')
            print(f"Successfully loaded dataset. Length: {len(ds)}")
            if len(ds) > 0:
                img, lbl = ds[0]
                print(f"Sample 0 - Image size: {img.size}, Label: {lbl} ({'PA' if lbl==1 else 'AP'})")
        except Exception as e:
            print(f"Test failed: {e}")
    else:
        print(f"CSV not found at {csv_path}")
