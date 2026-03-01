import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class COVIDxResolutionDataset(Dataset):
    def __init__(self, csv_file, root_dir, split='train', transform=None, use_resized=False):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.use_resized = use_resized
        
        # Filter by split
        self.data = self.data[self.data['split'] == split].reset_index(drop=True)
        
        # Label mapping (Negative=0, Positive=1)
        self.label_map = {'negative': 0, 'positive': 1}
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        filepath = row['filepath']
        
        if self.use_resized:
            # Path transformation: covidx/train/filename -> covidx/train_256/filename
            parts = filepath.split('/')
            if len(parts) >= 3:
                parts[1] = parts[1] + '_256'
                filepath = '/'.join(parts)
        
        img_path = os.path.join(self.root_dir, filepath)
        image = Image.open(img_path).convert('RGB')
        label = self.label_map[row['class']]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

