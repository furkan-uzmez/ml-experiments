import pytest
import sys
import os
import yaml
from unittest.mock import patch, mock_open, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import from the specific paths
from src.dataio.unet_dataset import get_unet_transforms

MOCK_UNET_CONFIG = """
preprocessing:
  target_size: [128, 128]
  mean: [0.5, 0.5, 0.5]
  std: [0.5, 0.5, 0.5]
augmentation:
  rand_rotate:
    prob: 0.0
  rand_flip_h:
    prob: 0.0
  rand_flip_v:
    prob: 0.0
  rand_zoom:
    prob: 0.0
"""

@patch("builtins.open", new_callable=mock_open, read_data=MOCK_UNET_CONFIG)
def test_get_unet_transforms(mock_file):
    """Verify that transformation pipelines are successfully instantiated."""
    # Should not throw any exceptions
    train_transforms = get_unet_transforms("dummy.yaml", is_train=True)
    val_transforms = get_unet_transforms("dummy.yaml", is_train=False)
    
    assert train_transforms is not None
    assert val_transforms is not None
