import yaml
from monai.networks.nets import BasicUNet
import torch.nn as nn

def create_unet_model(config_path: str = "configs/unet.yaml") -> nn.Module:
    """Creates a MONAI BasicUNet model based on the configuration file.
    
    Reads the specified YAML configuration and instantiates the MONAI
    BasicUNet architecture with the defined parameters.
    
    Args:
        config_path: Path to the U-Net configuration YAML file.
        
    Returns:
        An instantiated PyTorch nn.Module (MONAI BasicUNet).
        
    Raises:
        FileNotFoundError: If the configuration file is missing.
        KeyError: If required model parameters are missing in the config.
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    model_cfg = config['model']
    
    model = BasicUNet(
        spatial_dims=model_cfg.get('spatial_dims', 2),
        in_channels=model_cfg.get('in_channels', 3),
        out_channels=model_cfg.get('out_channels', 1),
        features=tuple(model_cfg.get('features', [32, 32, 64, 128, 256, 32])),
        dropout=model_cfg.get('dropout', 0.1)
    )
    
    return model
