import os
import sys
import json
import yaml
import shutil
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.dataio.dataset_index import DatasetIndex
from src.dataio.split_loader import SplitLoader


def convert_image_to_nnunet_format(case_dict: dict, out_images_dir: str, out_labels_dir: str, is_train: bool):
    """Converts a single ISIC 2018 case to nnU-Net's strict format.
    
    Reads the RGB image and splits it into three identical-size grayscale
    images, writing them as `<patient_id>_0000.png` (Red), `_0001.png` (Green),
    and `_0002.png` (Blue).
    
    Reads the ground truth mask and writes it as `<patient_id>.png` (values 0 and 1).
    """
    patient_id = case_dict['patient_id']
    img_path = case_dict['image_path']
    mask_path = case_dict['mask_path']
    
    # 1. Process RGB Image -> 3 modalities
    try:
        img = Image.open(img_path).convert('RGB')
        r, g, b = img.split()
        
        # Save each channel as a separate modality per nnUNet v2 requirements
        r.save(os.path.join(out_images_dir, f"{patient_id}_0000.png"))
        g.save(os.path.join(out_images_dir, f"{patient_id}_0001.png"))
        b.save(os.path.join(out_images_dir, f"{patient_id}_0002.png"))
        
    except Exception as e:
        print(f"Error processing image {img_path}: {e}")
        raise
        
    # 2. Process Label (Only if training/validation - nnU-Net test set lacks labels conceptually)
    # But for our benchmark, we place everything in imagesTr/labelsTr and control the 
    # splits using splits_final.json so nnU-Net validates natively. Let's just create labels 
    # for everything that has a ground truth.
    if is_train and os.path.exists(mask_path):
        try:
            mask = Image.open(mask_path).convert('L')
            mask_arr = np.array(mask)
            # Ensure background is 0, foreground is 1
            mask_arr = (mask_arr > 127).astype(np.uint8)
            mask_bin = Image.fromarray(mask_arr, mode='L')
            mask_bin.save(os.path.join(out_labels_dir, f"{patient_id}.png"))
        except Exception as e:
            print(f"Error processing mask {mask_path}: {e}")
            raise


def generate_dataset_json(out_dir: str, num_training_cases: int):
    """Generates the mandatory dataset.json for nnU-Net v2."""
    dataset_info = {
        "channel_names": {
            "0": "R",
            "1": "G",
            "2": "B"
        },
        "labels": {
            "background": 0,
            "lesion": 1
        },
        "numTraining": num_training_cases,
        "file_ending": ".png",
        "name": "Dataset500_ISIC2018",
        "description": "Medical Image Segmentation Benchmark - ISIC 2018",
        "reference": "ISIC Archive",
        "licence": "CC-BY",
        "release": "2018",
        "tensorImageSize": "2D"
    }
    
    with open(os.path.join(out_dir, "dataset.json"), 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, indent=4)


def main():
    """Main execution block for format conversion."""
    print("--- Starting ISIC 2018 to nnU-Net v2 Format Conversion ---")
    
    # Setup paths
    nnunet_raw = os.environ.get("nnUNet_raw", "artifacts/nnunet/nnUNet_raw")
    dataset_name = "Dataset500_ISIC2018"
    target_dataset_dir = os.path.join(nnunet_raw, dataset_name)
    
    imagesTr = os.path.join(target_dataset_dir, "imagesTr")
    labelsTr = os.path.join(target_dataset_dir, "labelsTr")
    
    # For evaluate-on-the-fly, nnU-Net needs the test images in a separate inference folder later
    # But for raw data formatting, `imagesTr` gets everything that has labels.
    
    os.makedirs(imagesTr, exist_ok=True)
    os.makedirs(labelsTr, exist_ok=True)
    
    # Load indexes
    index = DatasetIndex()
    split_loader = SplitLoader()
    
    train_ids = split_loader.get_train_ids()
    val_ids = split_loader.get_val_ids()
    test_ids = split_loader.get_test_ids()
    
    # We put train and val in imagesTr because nnUNet cross-validation handles both
    # Our custom split_loader dictates exactly what is train vs val inside splits_final.json
    # Test cases must go to a separate folder if they have no labels, but for the benchmark
    # ISIC 2018 train split actually contains labels. We will feed test cases natively
    # via the predict command later, so they don't strictly need to be in imagesTr UNLESS
    # we want nnUNet to check them. Let's process test_ids for inference later.
    
    # Merge train and val since nnUNet expects the full pool in imagesTr
    pool_ids = train_ids + val_ids
    
    print(f"Converting {len(pool_ids)} cases into imagesTr/labelsTr...")
    
    for pid in tqdm(pool_ids):
        case = index.get_case(pid)
        case['patient_id'] = pid
        convert_image_to_nnunet_format(case, imagesTr, labelsTr, is_train=True)
        
    generate_dataset_json(target_dataset_dir, num_training_cases=len(pool_ids))
    
    # Also prep test images for inference into a separate folder (not strictly nnUNet format, 
    # but same split channel logic applies for `nnUNetv2_predict`)
    imagesTs = os.path.join(target_dataset_dir, "imagesTs")
    os.makedirs(imagesTs, exist_ok=True)
    
    print(f"Converting {len(test_ids)} test cases into imagesTs...")
    for pid in tqdm(test_ids):
        case = index.get_case(pid)
        case['patient_id'] = pid
        # is_train=False means don't copy labels
        convert_image_to_nnunet_format(case, imagesTs, labelsTr, is_train=False)
        
    print(f"Conversion complete. raw data at: {target_dataset_dir}")


if __name__ == "__main__":
    main()
