import json
import os
from typing import List, Dict

class SplitLoader:
    """Loads and manages the deterministic patient-level data splits.
    
    This class ensures all model pipelines use the exact same train, validation,
    and test sets by providing centralized access to the split definition file.
    
    Attributes:
        split_path (str): Path to the primary split JSON file.
        split_data (Dict): The parsed split configuration.
    """
    
    def __init__(self, split_path: str = "splits/primary_split.json") -> None:
        """Initialize the split loader.
        
        Args:
            split_path: Internal path to the primary_split.json file.
            
        Raises:
            FileNotFoundError: If the split file cannot be found.
        """
        if not os.path.exists(split_path):
            raise FileNotFoundError(f"Split file not found: {split_path}")
            
        with open(split_path, 'r', encoding='utf-8') as f:
            self.split_data = json.load(f)
            
        # Optional: Validate split dictionary structure
        if not all(k in self.split_data for k in ['train', 'val', 'test']):
            raise KeyError("Split JSON must contain 'train', 'val', and 'test' keys.")

    def get_train_ids(self) -> List[str]:
        """Get the list of patient IDs assigned to the training set."""
        return self.split_data['train']

    def get_val_ids(self) -> List[str]:
        """Get the list of patient IDs assigned to the validation set."""
        return self.split_data['val']

    def get_test_ids(self) -> List[str]:
        """Get the list of patient IDs assigned to the test set."""
        return self.split_data['test']
        
    def verify_no_overlap(self) -> bool:
        """Verify that there is strictly zero overlap between predefined splits.
        
        Returns:
            bool: True if splits are mutually exclusive, False otherwise.
        """
        train_set = set(self.get_train_ids())
        val_set = set(self.get_val_ids())
        test_set = set(self.get_test_ids())
        
        overlap_tv = train_set.intersection(val_set)
        overlap_tt = train_set.intersection(test_set)
        overlap_vt = val_set.intersection(test_set)
        
        if overlap_tv or overlap_tt or overlap_vt:
            return False
            
        return True
