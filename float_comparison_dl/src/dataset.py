import torch
from torch.utils.data import Dataset, DataLoader

class DummyMedicalDataset(Dataset):
    "A dummy dataset representing high-resolution medical images with 2 classes (e.g., Normal vs Pneumonia)"
    def __init__(self, num_samples, img_size=224, num_classes=2, imbalance_ratio=0.8):
        super().__init__()
        self.num_samples = num_samples
        self.img_size = img_size
        self.num_classes = num_classes
        
        # Create dummy images (C, H, W)
        self.data = torch.randn(num_samples, 3, img_size, img_size)
        
        # Create imbalanced labels
        num_majority = int(num_samples * imbalance_ratio)
        labels_maj = torch.zeros(num_majority, dtype=torch.long)
        labels_min = torch.ones(num_samples - num_majority, dtype=torch.long)
        self.targets = torch.cat([labels_maj, labels_min])
        
        # Shuffle
        indices = torch.randperm(num_samples)
        self.data = self.data[indices]
        self.targets = self.targets[indices]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

def get_dataloaders(batch_size=32, num_workers=4):
    train_ds = DummyMedicalDataset(num_samples=1000)
    val_ds = DummyMedicalDataset(num_samples=200)
    test_ds = DummyMedicalDataset(num_samples=200)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader, test_loader
