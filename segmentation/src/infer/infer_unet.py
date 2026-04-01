import os
import sys

import numpy as np
import torch
import yaml
from PIL import Image
from tqdm import tqdm
import argparse
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.models.unet_model import create_unet_model
from src.dataio.unet_dataset import create_datasets
from src.dataio.dataset_index import DatasetIndex
from src.evaluation.resample import resample_to_reference
from src.evaluation.runtime import track_inference_time, track_peak_gpu_memory
from src.reporting.io_utils import write_jsonl_log
from src.reporting.reporting_contract import RUNTIME_LOG_FILENAME

# For loading the original reference image dimensions during resampling
import SimpleITK as sitk

@track_peak_gpu_memory
@track_inference_time
def infer_single_batch(model: torch.nn.Module, inputs: torch.Tensor) -> dict:
    """Performs inference on a single batch, tracked by runtime decorators."""
    with torch.no_grad():
        outputs = model(inputs)
        # Apply sigmoid to convert raw logits to probabilities
        probs = torch.sigmoid(outputs)
    return {"probs": probs}

def infer_unet(seed: int) -> None:
    """Runs inference over the test set for a given trained U-Net seed.
    
    Loads the best weights, performs feed-forward on test cases, tracks
    latency/memory, applies native resolution resampling, and saves the
    binary predictions to the artifacts folder.
    
    Args:
        seed: The integer seed of the trained model to evaluate.
    """
    with open("configs/unet.yaml", 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Inference Seed {seed} on device {device} ---")
    
    # Checkpoint and output setup
    checkpoint_path = os.path.join(config['training']['save_dir'], f"seed_{seed}", "best_model.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
    out_dir = os.path.join(config['inference']['output_dir'], f"seed_{seed}")
    os.makedirs(out_dir, exist_ok=True)
    
    # Model
    model = create_unet_model().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    # Dataset (test set only)
    # We use batch_size 1 for inference so we can accurately measure per-case latency
    # and safely resample arbitrarily sized images.
    _, _, test_ds = create_datasets()
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    
    # Need access to original native resolution references
    dataset_index = DatasetIndex()
    
    resample = config['inference'].get('resample_to_native', True)
    
    # Tracking
    inference_times = []
    peak_memories = []
    runtime_rows: list[dict] = []
    
    pbar = tqdm(test_loader, desc=f"Inference Seed {seed}")
    for batch_data in pbar:
        # In batch_size=1, inputs is (1, 3, H, W)
        inputs = batch_data["image"].to(device)
        patient_id = batch_data["patient_id"][0]
        
        # Tracked inference
        result = infer_single_batch(model, inputs)
        
        probs = result["probs"].cpu().numpy()  # (1, 1, H, W)
        bin_mask = (probs[0, 0] > 0.5).astype(np.uint8)
        
        # Tracking metrics collection
        inference_times.append(result['inference_time_seconds'])
        peak_memories.append(result['peak_gpu_memory_mb'])
        runtime_rows.append(
            {
                "run_id": f"unet_seed_{seed}",
                "case_id": patient_id,
                "inference_time_seconds": result["inference_time_seconds"],
                "peak_gpu_memory_mb": result["peak_gpu_memory_mb"],
            }
        )
        
        # Resampling
        if resample:
            # Load original mask to get native dimensions
            ref_path = dataset_index.get_case(patient_id)["mask_path"]
            ref_img = sitk.ReadImage(ref_path)
            
            # Use Phase 2 resampler (nearest neighbor for binary)
            final_mask = resample_to_reference(bin_mask, ref_img, is_binary=True)
        else:
            final_mask = bin_mask
            
        # Save prediction
        # Map [0, 1] back to [0, 255] for standard ground truth representation
        img = Image.fromarray(final_mask * 255, mode='L')
        save_path = os.path.join(out_dir, f"{patient_id}_segmentation.png")
        img.save(save_path)

    write_jsonl_log(os.path.join(out_dir, RUNTIME_LOG_FILENAME), runtime_rows)
        
    avg_latency = float(np.mean(inference_times))
    max_memory = float(np.max(peak_memories))
    
    print(f"Inference Complete. Saved {len(test_ds)} masks to {out_dir}")
    print(f"Average Inference Time: {avg_latency:.4f} sec/case")
    print(f"Max Peak GPU Memory: {max_memory:.2f} MB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Infer U-Net Baseline")
    parser.add_argument("--seed", type=int, required=True, help="Random seed for this run")
    args = parser.parse_args()
    
    infer_unet(args.seed)
