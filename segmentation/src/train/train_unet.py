import os
import sys
import yaml
import torch
from torch.utils.data import DataLoader
from monai.losses import DiceFocalLoss
from tqdm import tqdm
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.models.unet_model import create_unet_model
from src.dataio.unet_dataset import create_datasets
from src.evaluation.metrics import compute_metrics


def set_deterministic_seeds(seed: int, config_path: str = "configs/seeds.yaml") -> None:
    """Sets random seeds across Python, NumPy, and PyTorch for reproducibility."""
    import random
    import numpy as np
    
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)['seeds']
        
    if cfg.get('set_python_random', True):
        random.seed(seed)
    if cfg.get('set_numpy', True):
        np.random.seed(seed)
    if cfg.get('set_torch', True):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            if cfg.get('set_cuda_deterministic', True):
                torch.cuda.manual_seed_all(seed)
            if cfg.get('cudnn_deterministic', True):
                torch.backends.cudnn.deterministic = True
            if not cfg.get('cudnn_benchmark', False):
                torch.backends.cudnn.benchmark = False


def train_unet(seed: int) -> None:
    """Trains a U-Net model from scratch for a single specified seed.
    
    Loads configuration from `configs/unet.yaml`, sets deterministic seeds,
    initializes the dataset, model, loss, and optimizer, and executes the
    training loop with validation at each epoch. Saves the best model weights.
    
    Args:
        seed: The integer seed for this training run.
    """
    set_deterministic_seeds(seed)
    
    with open("configs/unet.yaml", 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    train_cfg = config['training']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Training Seed {seed} on device {device} ---")
    
    # Checkpoint dir
    save_dir = os.path.join(train_cfg['save_dir'], f"seed_{seed}")
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, "best_model.pth")
    
    # 1. Dataset & DataLoader
    train_ds, val_ds, _ = create_datasets()
    
    # Use standard PyTorch DataLoader; MONAI Dataset inherits from standard PyTorch
    train_loader = DataLoader(
        train_ds, 
        batch_size=train_cfg['batch_size'], 
        shuffle=True, 
        num_workers=train_cfg['num_workers'],
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=train_cfg['batch_size'], 
        shuffle=False, 
        num_workers=train_cfg['num_workers'],
        pin_memory=torch.cuda.is_available()
    )
    
    # 2. Model, Loss, Optimizer
    model = create_unet_model().to(device)
    
    loss_cfg = config['loss']
    criterion = DiceFocalLoss(
        include_background=loss_cfg.get('include_background', False),
        sigmoid=loss_cfg.get('sigmoid', True),
        squared_pred=loss_cfg.get('squared_pred', True)
    )
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=train_cfg['learning_rate'], 
        weight_decay=train_cfg.get('weight_decay', 1e-5)
    )
    
    # 3. Training Loop
    best_val_dice = -1.0
    epochs = train_cfg['epochs']
    
    # Initialize CSV Log
    log_file = os.path.join(save_dir, "training_log.csv")
    with open(log_file, "w") as f:
        f.write("epoch,train_loss,val_loss,val_dice\\n")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        # tqdm for training batches
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch_data in pbar:
            inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_dice_sum = 0.0
        
        print(f"Running validation...")
        with torch.no_grad():
            for batch_data in val_loader:
                inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                # Binarize outputs for metric calculation
                # Sigmoid because out_channels=1 without final activation (MONAI default)
                outputs_prob = torch.sigmoid(outputs)
                outputs_bin = (outputs_prob > 0.5).cpu().numpy()
                labels_bin = labels.cpu().numpy()
                
                # Compute batch dice manually (not full eval stack, just for model selection)
                for b in range(outputs_bin.shape[0]):
                    metrics = compute_metrics(outputs_bin[b, 0], labels_bin[b, 0])
                    val_dice_sum += metrics['dice']
                    
        val_loss /= len(val_loader)
        avg_val_dice = val_dice_sum / len(val_ds)
        
        # Save to CSV log
        with open(log_file, "a") as f:
            f.write(f"{epoch+1},{train_loss:.4f},{val_loss:.4f},{avg_val_dice:.4f}\\n")
            
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Dice: {avg_val_dice:.4f}")
        
        # Model Selection
        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved new best model with Val Dice: {best_val_dice:.4f}")

    print(f"Training completed for Seed {seed}. Best Val Dice: {best_val_dice:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train U-Net Baseline")
    parser.add_argument("--seed", type=int, required=True, help="Random seed for this run")
    args = parser.parse_args()
    
    train_unet(args.seed)
