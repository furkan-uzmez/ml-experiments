import json
import os
import sys
from datetime import datetime, timezone

import torch
import yaml
from monai.data import DataLoader
from monai.losses import DiceFocalLoss
from tqdm import tqdm
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.models.unet_model import create_unet_model
from src.dataio.unet_dataset import create_datasets
from src.evaluation.metrics import compute_metrics
from src.reporting.io_utils import write_json
from src.reporting.reporting_contract import TRAINING_HISTORY_FILENAME


def set_deterministic_seeds(seed: int, config_path: str = "configs/seeds.yaml") -> None:
    """Sets random seeds across Python, NumPy, and PyTorch for reproducibility."""
    import random
    import numpy as np
    from monai.utils import set_determinism
    
    set_determinism(seed=seed)
    
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


def compact_config_summary(config: dict) -> str:
    return json.dumps(config, sort_keys=True, separators=(",", ":"))


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
    history_path = os.path.join(save_dir, TRAINING_HISTORY_FILENAME)
    manifest_path = os.path.join(save_dir, "run_manifest.json")
    train_started_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    
    # 1. Dataset & DataLoader
    train_ds, val_ds, _ = create_datasets()
    
    # Use MONAI DataLoader to ensure deterministic seeds across multiprocessing workers
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
        lr=float(train_cfg['learning_rate']), 
        weight_decay=float(train_cfg.get('weight_decay', 1e-5))
    )
    
    # 3. Training Loop
    best_val_dice = -1.0
    epochs = train_cfg['epochs']
    history_rows: list[dict] = []
    
    # Initialize Logger
    log_file = os.path.join(save_dir, "training_log.log")
    with open(log_file, "w") as f:
        f.write("=== U-Net Benchmark Training Log ===\n")
        f.write(f"Seed: {seed}\n")
        f.write("------------------------------------\n")
    
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
        
        # Save to plain text log
        log_line = f"Epoch {epoch+1:03d}/{epochs:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Dice: {avg_val_dice:.4f}"
        with open(log_file, "a") as f:
            f.write(log_line + "\n")

        history_rows.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_dice": avg_val_dice,
            }
        )
            
        print(log_line)
        
        # Model Selection
        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved new best model with Val Dice: {best_val_dice:.4f}")

    train_finished_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    write_json(history_path, history_rows)

    run_manifest = {
        "run_id": f"unet_seed_{seed}",
        "model_name": "U-Net",
        "backbone": config["model"].get("architecture", "BasicUNet"),
        "seed": seed,
        "loss_name": loss_cfg.get("name", "DiceFocalLoss"),
        "optimizer": config.get("optimizer", {}).get("name", "AdamW"),
        "learning_rate": train_cfg.get("learning_rate"),
        "scheduler": train_cfg.get("scheduler", "none"),
        "augmentation_summary": compact_config_summary(config.get("augmentation", {})),
        "preprocessing_summary": compact_config_summary(config.get("preprocessing", {})),
        "threshold_policy": "fixed_threshold@0.5",
        "checkpoint_path": best_model_path,
        "train_started_at": train_started_at,
        "train_finished_at": train_finished_at,
        "history_path": history_path,
        "log_path": log_file,
        "best_val_dice": best_val_dice,
    }
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(run_manifest, handle, indent=2)

    print(f"Training completed for Seed {seed}. Best Val Dice: {best_val_dice:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train U-Net Baseline")
    parser.add_argument("--seed", type=int, required=True, help="Random seed for this run")
    args = parser.parse_args()
    
    train_unet(args.seed)
