#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import timm
import os
import sys
import gc
import random
import numpy as np

# Add functions directory to path to import local modules
current_dir = os.path.dirname(os.path.abspath(__file__))
functions_path = os.path.join(current_dir, "functions")
if functions_path not in sys.path:
    sys.path.append(functions_path)

from dataset import COVIDCXNetDataset
from train import train

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

import argparse

from logger import get_logger

def run_batch_size_experiments(image_path, batch_sizes=None, num_epochs=5, device=None, project_root=None, model_name='resnet50', train_ds=None, val_ds=None, save_dir=None, log_dir=None):
    # Configuration
    # NOTE: Please verify these paths match your environment
    PROJECT_ROOT = project_root if project_root else current_dir
    
    if batch_sizes is None:
        batch_sizes = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    set_seed(42)
    
    results = {}

    # Aggressive cleanup before starting experiments
    gc.collect()
    torch.cuda.empty_cache()

    for bs in batch_sizes:
        print(f"\n{'='*30}\nRunning experiment with Batch Size: {bs}\n{'='*30}")
        
        try:
            train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=4, pin_memory=True)
            val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)
            
            model = timm.create_model(model_name, pretrained=True)
            model.reset_classifier(num_classes=2)
            model.to(device)
            
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            # Prepare paths
            current_save_dir = save_dir if save_dir else os.path.join(PROJECT_ROOT, 'models')
            current_log_dir = log_dir if log_dir else os.path.join(PROJECT_ROOT, 'logs')
            
            os.makedirs(current_save_dir, exist_ok=True)
            os.makedirs(current_log_dir, exist_ok=True)

            save_path = os.path.join(current_save_dir, f'{model_name}_bs{bs}.pth')
            log_path = os.path.join(current_log_dir, f'{model_name}_bs{bs}.log')
            
            # Initialize logger and log dataset stats
            logger = get_logger(log_path)
            
            def log_dataset_stats(ds, name):
                try:
                    size = len(ds)
                    logger.info(f"{name} Dataset Size: {size}")
                    if hasattr(ds, 'data') and 'projection' in ds.data.columns:
                        ap_count = len(ds.data[ds.data['projection'] == 'AP'])
                        pa_count = len(ds.data[ds.data['projection'] == 'PA'])
                        logger.info(f"{name} Dataset Stats: AP={ap_count}, PA={pa_count}")
                except Exception as e:
                    logger.warning(f"Could not log stats for {name}: {e}")

            log_dataset_stats(train_ds, "Train")
            log_dataset_stats(val_ds, "Val")

            # Train
            train_losses, train_accs, val_losses, val_accs = train(
                model, train_loader, val_loader, criterion, optimizer, device,
                save_path=save_path, num_epochs=num_epochs, patience=3, log_path=log_path
            )

            # Store results
            results[bs] = max(val_accs)
            print(f"Best Val Acc for BS {bs}: {results[bs]:.4f}")
            
        except torch.cuda.OutOfMemoryError:
            print(f"CRITICAL: Out of Memory for Batch Size {bs}. Skipping higher batch sizes.")
            break
        except Exception as e:
            print(f"An error occurred with Batch Size {bs}: {e}")
        finally:
            # Cleanup memory
            mem_before = torch.cuda.memory_reserved() if torch.cuda.is_available() else 0
            
            # Delete variables if they exist
            # We use locals() in the original script loop, but here variables are local to the function
            # We can use del explicitly or rely on scope, but explicit delete is safer for immediate cleanup
            if 'model' in locals(): del model
            if 'optimizer' in locals(): del optimizer
            if 'train_loader' in locals(): del train_loader
            if 'val_loader' in locals(): del val_loader
            # Do NOT delete train_ds/val_ds if they were passed in, but we might need to handle memory.
            # For now, let's just delete the loaders and model as they use GPU memory mostly.
            # And gc.collect() handles the rest.
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                mem_after = torch.cuda.memory_reserved()
                cleared_mb = (mem_before - mem_after) / (1024 ** 2)
                print(f"Memory cleared after BS {bs}: {cleared_mb:.2f} MB.")
            else:
                print(f"Memory cleared after BS {bs}.")
    
    print("\n--- Final Results ---")
    for bs, res in results.items():
        print(f"Batch Size {bs}: {res}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Batch Size Experiments for COVIDCXNet")
    parser.add_argument("--image_path", type=str, default=r'E:\covidx', help="Path to the image dataset")
    parser.add_argument("--batch_sizes", type=int, nargs='+', default=[8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192], help="List of batch sizes to experiment with")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs per experiment")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda or cpu)")
    parser.add_argument("--project_root", type=str, default=None, help="Project root directory for saving models/logs")
    parser.add_argument("--model", type=str, default='resnet50', help="Model architecture to use")
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save models")
    parser.add_argument("--log_dir", type=str, default=None, help="Directory to save logs")

    args = parser.parse_args()
    
    device = torch.device(args.device) if args.device else None
    
    run_batch_size_experiments(args.image_path, args.batch_sizes, args.epochs, device, args.project_root, args.model, save_dir=args.save_dir, log_dir=args.log_dir)

