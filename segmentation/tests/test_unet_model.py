import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.unet_model import create_unet_model

def test_unet_output_shape():
    """Verify that the BasicUNet produces the exact expected output shape."""
    model = create_unet_model(config_path="configs/unet.yaml")
    model.eval()
    
    # 2D ISIC images: Batch of 2, 3 channels (RGB), 256x256 resolution
    dummy_input = torch.randn(2, 3, 256, 256)
    
    with torch.no_grad():
        output = model(dummy_input)
        
    # Expected output: Batch of 2, 1 channel (Binary Mask logit), 256x256 resolution
    assert output.shape == (2, 1, 256, 256)
