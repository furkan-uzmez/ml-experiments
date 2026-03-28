import os
import sys
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.dataio.split_loader import SplitLoader

def enforce_nnunet_split():
    """Generates splits_final.json to force nnU-Net to use the predefined deterministic split.
    
    nnU-Net defaults to 5-fold cross-validation. By generating `splits_final.json` in the
    `nnUNet_preprocessed` directory, we bypass the random fold generator and ensure the
    val split matches the benchmark exactly. We use Fold 0 as our deterministic split.
    """
    nnunet_preprocessed = os.environ.get("nnUNet_preprocessed", "artifacts/nnunet/nnUNet_preprocessed")
    dataset_name = "Dataset500_ISIC2018"
    target_dir = os.path.join(nnunet_preprocessed, dataset_name)
    
    if not os.path.exists(target_dir):
        raise FileNotFoundError(f"Preprocessed directory not found: {target_dir}. Did you run nnUNetv2_plan_and_preprocess?")
        
    split_loader = SplitLoader()
    
    # nnU-Net splits_final format is a list of dictionaries per fold
    # [{"train": ["case_1", "case_2"], "val": ["case_3"]}, ...]
    # We will provide exactly 1 split corresponding to 'fold 0'
    
    split_data = [{
        "train": split_loader.get_train_ids(),
        "val": split_loader.get_val_ids()
    }]
    
    out_file = os.path.join(target_dir, "splits_final.json")
    
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(split_data, f, indent=4)
        
    print(f"Successfully enforced custom deterministic splits via {out_file}")
    print(f"Fold 0 -> Train: {len(split_data[0]['train'])}, Val: {len(split_data[0]['val'])}")

if __name__ == "__main__":
    enforce_nnunet_split()
