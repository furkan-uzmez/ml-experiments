
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import numpy as np
import gc
from tqdm import tqdm
import timm

# Add functions directory to path to import local modules
current_dir = os.path.dirname(os.path.abspath(__file__))
functions_path = os.path.join(current_dir)
if functions_path not in sys.path:
    sys.path.append(functions_path)

from dataset import COVIDCXNetDataset

def run_sample_experiment(image_path, batch_sizes=None, sample_train_size=2000, sample_val_size=500, model_name='resnet50', device=None, seed=42):
    """
    Runs a batch size experiment on a small subset of the dataset to estimate memory usage and feasibility.
    
    Args:
        image_path (str): Path to the image dataset folder (containing 'covidx_merged.csv' or one level above).
        batch_sizes (list, optional): List of batch sizes to test. Defaults to [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192].
        sample_train_size (int): Number of training samples to use for the experiment.
        sample_val_size (int): Number of validation samples to use.
        model_name (str): Name of the model to use (from timm).
        device (torch.device, optional): Device to run the experiment on.
        seed (int): Random seed for reproducibility.
    
    Returns:
        dict: A dictionary mapping batch sizes to success/error messages.
    """
    if batch_sizes is None:
        batch_sizes = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
        
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Device: {device}")
    
    # --- Dataset Setup ---
    print("Setting up dataset...")
    
    TRAIN_CSV = os.path.join(image_path, 'covidx_merged.csv')
    ROOT_DATA_DIR = os.path.dirname(image_path) # Assuming image_path is where covidx_merged.csv is, and images are relative or in parent?
    # Based on previous context: IMAGE_PATH was r'E:\covidx' and ROOT_DATA_DIR was dirname(IMAGE_PATH).
    # dataset.py logic: root_dir is parent of 'covidx' if filepath starts with 'covidx/'.
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Load Train and Val Datasets
    # If size is 'full', use the respective split directly. 
    # Otherwise, take a random subset of that split.
    
    train_dataset = COVIDCXNetDataset(csv_file=TRAIN_CSV, root_dir=ROOT_DATA_DIR, transform=transform, split='train')
    if sample_train_size == 'full':
        train_subset = train_dataset
    else:
        indices_train = list(range(len(train_dataset)))
        np.random.seed(seed)
        np.random.shuffle(indices_train)
        
        try:
            train_size_val = float(sample_train_size)
        except ValueError:
            train_size_val = int(sample_train_size)
            
        if isinstance(train_size_val, float) and 0.0 < train_size_val < 1.0:
            sample_train_size_int = int(len(train_dataset) * train_size_val)
        else:
            sample_train_size_int = min(int(train_size_val), len(indices_train))
            
        train_subset = Subset(train_dataset, indices_train[:sample_train_size_int])

    val_dataset = COVIDCXNetDataset(csv_file=TRAIN_CSV, root_dir=ROOT_DATA_DIR, transform=transform, split='val')
    if sample_val_size == 'full':
        val_subset = val_dataset
    else:
        indices_val = list(range(len(val_dataset)))
        np.random.seed(seed)
        np.random.shuffle(indices_val)
        
        try:
            val_size_val = float(sample_val_size)
        except ValueError:
            val_size_val = int(sample_val_size)
            
        if isinstance(val_size_val, float) and 0.0 < val_size_val < 1.0:
            sample_val_size_int = int(len(val_dataset) * val_size_val)
        else:
            sample_val_size_int = min(int(val_size_val), len(indices_val))
            
        val_subset = Subset(val_dataset, indices_val[:sample_val_size_int])

    print(f"Sample Train Size: {len(train_subset)}")
    print(f"Sample Val Size: {len(val_subset)}")

    results = {}

    for bs in batch_sizes:
        print(f"\n--- Testing Batch Size: {bs} ---")
        try:
            # 1. Clear Memory
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            # 2. DataLoaders
            train_loader = DataLoader(train_subset, batch_size=bs, shuffle=True, num_workers=4)
            val_loader = DataLoader(val_subset, batch_size=bs, shuffle=False, num_workers=4)

            model = timm.create_model(model_name, pretrained=True)
            model.reset_classifier(num_classes=2)
            model.to(device)

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            # 4. Run One Epoch of Training
            model.train()
            # Wrap tqdm to avoid too much spam, leave=False
            for images, labels in tqdm(train_loader, desc=f"BS {bs} Train", leave=False):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            # 5. Validation (Optional, just to ensure inference works too)
            model.eval()
            with torch.no_grad():
                 for images, labels in val_loader:
                     images, labels = images.to(device), labels.to(device)
                     outputs = model(images)

            # Success
            max_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
            print(f"Success! Max Memory: {max_mem:.2f} MB")
            results[bs] = f"Success (Mem: {max_mem:.2f} MB)"

        except RuntimeError as e:
            if 'out of memory' in str(e):
                print(f"OOM Error for Batch Size {bs}")
                results[bs] = "OOM"
                torch.cuda.empty_cache() # Emergency cleanup
            else:
                print(f"RuntimeError: {e}")
                results[bs] = f"Error: {e}"
        except Exception as e:
            print(f"General Error: {e}")
            results[bs] = f"Error: {e}"
        finally:
            if 'model' in locals(): del model
            if 'optimizer' in locals(): del optimizer
            gc.collect()
            torch.cuda.empty_cache()

    print("\n--- Final Results ---")
    for bs, res in results.items():
        print(f"Batch Size {bs}: {res}")
    
    return results

if __name__ == "__main__":
    # Allow running from CLI
    import argparse
    parser = argparse.ArgumentParser(description="Run Sample Batch Size Experiment")
    parser.add_argument("--image_path", type=str, required=True, help="Path to image dataset")
    parser.add_argument("--batch_sizes", type=int, nargs='+', help="Batch sizes to test")
    parser.add_argument("--model", type=str, default="resnet50", help="Model name")
    
    def float_int_or_full(value):
        if value.lower() == 'full':
            return 'full'
        try:
            val = float(value)
            return int(val) if val.is_integer() else val
        except ValueError:
            raise argparse.ArgumentTypeError(f"Value must be a number or 'full', got '{value}'")

    parser.add_argument("--sample_train_size", type=float_int_or_full, default=2000, help="Number of training samples, fraction, or 'full'")
    parser.add_argument("--sample_val_size", type=float_int_or_full, default=500, help="Number of validation samples, fraction, or 'full'")
    
    args = parser.parse_args()
    
    run_sample_experiment(
        args.image_path, 
        batch_sizes=args.batch_sizes, 
        sample_train_size=args.sample_train_size,
        sample_val_size=args.sample_val_size,
        model_name=args.model
    )
