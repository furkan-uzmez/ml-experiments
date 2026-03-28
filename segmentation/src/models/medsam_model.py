import os
import urllib.request
import torch
from segment_anything import sam_model_registry, SamPredictor

class MedSAMModel:
    """Wrapper for the MedSAM (Segment Anything for Medical) Foundation Model.
    
    Handles downloading the weights securely (if missing), instantiating the 
    underlying SAM ViT-B architecture, and providing a clean prediction interface
    using the official SamPredictor wrapper.
    """
    
    def __init__(self, config: dict, device: str = "cuda"):
        """Initializes the MedSAM wrapper securely.
        
        Args:
            config: The parsed subset of `configs/medsam.yaml`['model'].
            device: Device to place the model on ('cuda' or 'cpu').
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_type = config.get("type", "vit_b")
        
        self.weight_dir = config.get("checkpoint_dir", "artifacts/medsam/weights")
        self.weight_name = config.get("checkpoint_name", "medsam_vit_b.pth")
        self.weight_path = os.path.join(self.weight_dir, self.weight_name)
        
        os.makedirs(self.weight_dir, exist_ok=True)
        
        # Download strictly required weights if they do not exist
        if not os.path.exists(self.weight_path):
            self._download_weights(config.get("download_url"))
            
        print(f"Loading MedSAM weights from {self.weight_path}...")
        self.sam_model = sam_model_registry[self.model_type](checkpoint=self.weight_path)
        self.sam_model = self.sam_model.to(self.device)
        self.predictor = SamPredictor(self.sam_model)
        
    def _download_weights(self, url: str):
        """Downloads weights using standard library with progress indication."""
        if not url:
            raise ValueError(f"Checkpoint not found at {self.weight_path} and no download URL provided.")
            
        print(f"Downloading MedSAM Checkpoint ({url}). This may take a few minutes...")
        
        def _progress(count, block_size, total_size):
            percent = int(count * block_size * 100 / total_size)
            print(f"\rDownloading... {percent}%", end="")
            
        urllib.request.urlretrieve(url, self.weight_path, reporthook=_progress)
        print("\nDownload complete.")
        
    def set_image(self, image_rgb: np.ndarray):
        """Passes the image to the SAM predictor to compute heavy image embeddings.
        
        Args:
            image_rgb: A HxWxC np.ndarray in RGB format (0-255).
        """
        import numpy as np
        # SAM expects the RGB image uint8 format internally
        self.predictor.set_image(image_rgb)
        
    @torch.no_grad()
    def predict(self, bbox: np.ndarray) -> np.ndarray:
        """Runs the MedSAM prompt prediction over the already-set image embeddings.
        
        Args:
            bbox: A 1D array/list [x_min, y_min, x_max, y_max].
            
        Returns:
            A binary boolean mask array (HxW) matching the originally set image dims.
        """
        import numpy as np
        # SAM returns multiple mask variants if multimask_output=True,
        # but the standard prompt approach yields a single consolidated prediction
        # when multimask_output=False.
        masks, _, _ = self.predictor.predict(
            box=np.array(bbox),
            multimask_output=False
        )
        return masks[0]
