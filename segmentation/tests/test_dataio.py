import pytest
from unittest.mock import patch, mock_open
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dataio.split_loader import SplitLoader
from src.dataio.dataset_index import DatasetIndex

MOCK_SPLIT_JSON = """
{
  "metadata": {"dataset": "TEST"},
  "train": ["P001", "P002", "P003"],
  "val": ["P004"],
  "test": ["P005"]
}
"""

@patch("os.path.exists", return_value=True)
@patch("builtins.open", new_callable=mock_open, read_data=MOCK_SPLIT_JSON)
def test_split_loader(mock_file, mock_exists):
    loader = SplitLoader()
    
    assert loader.get_train_ids() == ["P001", "P002", "P003"]
    assert loader.get_val_ids() == ["P004"]
    assert loader.get_test_ids() == ["P005"]
    assert loader.verify_no_overlap() is True


MOCK_OVERLAP_JSON = """
{
  "metadata": {"dataset": "TEST"},
  "train": ["P001", "P002"],
  "val": ["P002"],
  "test": ["P003"]
}
"""

@patch("os.path.exists", return_value=True)
@patch("builtins.open", new_callable=mock_open, read_data=MOCK_OVERLAP_JSON)
def test_split_loader_overlap(mock_file, mock_exists):
    loader = SplitLoader()
    assert loader.verify_no_overlap() is False


MOCK_CONFIG_YAML = """
dataset:
  root_path: "/dummy"
  images_dir: "images"
  masks_dir: "masks"
  image_extension: ".jpg"
  mask_extension: ".png"
"""

@patch("os.path.exists", return_value=True)
@patch("builtins.open", new_callable=mock_open, read_data=MOCK_CONFIG_YAML)
def test_dataset_index(mock_file, mock_exists):
    # Mocking os.path.exists everywhere, so it assumes images and masks exist
    index = DatasetIndex()
    
    case = index.get_case("P001")
    assert case["image_path"] == "/dummy/images/P001.jpg"
    assert case["mask_path"] == "/dummy/masks/P001.png"
