from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as functional
import yaml
from PIL import Image

DEFAULT_TEXT_PROMPT = "lesion"
DEFAULT_RESOLUTION = 1008
DEFAULT_REPO_URL = "https://github.com/Joey-S-Liu/MedSAM3"
SEGMENTATION_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPO_PATH = SEGMENTATION_ROOT / "external" / "MedSAM3"


@dataclass(frozen=True)
class MedSAM3Paths:
    repo_path: Path
    repo_config_path: Path
    weights_path: Path
    base_checkpoint_path: Path | None
    bpe_path: Path


def resolve_repo_relative_path(
    path_value: str | None,
    repo_path: Path,
    default_path: Path | None = None,
) -> Path | None:
    """Resolve repo-relative paths from the MedSAM3 config."""
    if path_value in (None, ""):
        return default_path

    candidate = Path(path_value).expanduser()
    if not candidate.is_absolute():
        candidate = repo_path / candidate
    return candidate.resolve()


def resolve_project_relative_path(
    path_value: str | None,
    project_root: Path,
    default_path: Path | None = None,
) -> Path | None:
    """Resolve project-relative paths from the segmentation workspace."""
    if path_value in (None, ""):
        candidate = default_path
    else:
        candidate = Path(path_value).expanduser()
        if not candidate.is_absolute():
            candidate = project_root / candidate

    if candidate is None:
        return None
    return candidate.resolve()


def normalize_text_prompts(
    raw_prompts: Sequence[str] | str | None,
    fallback_prompt: str = DEFAULT_TEXT_PROMPT,
) -> list[str]:
    """Normalize MedSAM3 prompt inputs into a non-empty prompt list."""
    if isinstance(raw_prompts, str):
        prompt_candidates = [raw_prompts]
    else:
        prompt_candidates = list(raw_prompts or [])

    prompts = [str(prompt).strip() for prompt in prompt_candidates if str(prompt).strip()]
    if prompts:
        return prompts

    fallback = fallback_prompt.strip()
    if not fallback:
        raise ValueError("At least one non-empty MedSAM3 text prompt is required.")
    return [fallback]


def combine_binary_masks(
    masks: Sequence[np.ndarray] | None,
    image_shape: tuple[int, int],
) -> np.ndarray:
    """Merge a sequence of binary masks into a single binary mask."""
    combined_mask = np.zeros(image_shape, dtype=np.uint8)
    if not masks:
        return combined_mask

    for mask in masks:
        if mask.shape != image_shape:
            raise ValueError(
                f"Mask shape {mask.shape} does not match target image shape {image_shape}."
            )
        combined_mask |= mask.astype(np.uint8)

    return combined_mask


class MedSAM3Model:
    """Local wrapper around the MedSAM3 repo for text-guided segmentation."""

    def __init__(self, config: dict[str, Any], device: str = "cuda") -> None:
        self.repo_url = str(config.get("repo_url") or DEFAULT_REPO_URL).strip()
        self.repo_path = self._resolve_repo_path(config)
        self.repo_config_path = resolve_repo_relative_path(
            config.get("config_path"),
            self.repo_path,
            self.repo_path / "configs" / "full_lora_config.yaml",
        )
        if self.repo_config_path is None or not self.repo_config_path.exists():
            raise FileNotFoundError(
                f"MedSAM3 config not found: {self.repo_config_path}"
            )

        self.repo_config = self._read_repo_config(self.repo_config_path)
        self.paths = self._resolve_paths(config)
        self.prompts = normalize_text_prompts(
            raw_prompts=config.get("text_prompts"),
            fallback_prompt=config.get("default_prompt", DEFAULT_TEXT_PROMPT),
        )
        self.resolution = int(config.get("resolution", DEFAULT_RESOLUTION))
        self.detection_threshold = float(config.get("detection_threshold", 0.5))
        self.nms_iou_threshold = float(config.get("nms_iou_threshold", 0.5))
        self.mask_threshold = float(config.get("mask_threshold", 0.5))
        self.load_from_hf = bool(config.get("load_from_hf", True))
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self._ensure_repo_on_sys_path()
        runtime_modules = self._import_runtime_modules()
        self._build_sam3_image_model = runtime_modules["build_sam3_image_model"]
        self._datapoint_type = runtime_modules["Datapoint"]
        self._sam_image_type = runtime_modules["SAMImage"]
        self._find_query_type = runtime_modules["FindQueryLoaded"]
        self._inference_metadata_type = runtime_modules["InferenceMetadata"]
        self._collate_fn_api = runtime_modules["collate_fn_api"]
        self._copy_data_to_device = runtime_modules["copy_data_to_device"]
        self._compose_api = runtime_modules["ComposeAPI"]
        self._random_resize_api = runtime_modules["RandomResizeAPI"]
        self._to_tensor_api = runtime_modules["ToTensorAPI"]
        self._normalize_api = runtime_modules["NormalizeAPI"]
        self._nms = runtime_modules["nms"]
        self._lora_config_type = runtime_modules["LoRAConfig"]
        self._apply_lora_to_model = runtime_modules["apply_lora_to_model"]
        self._load_lora_weights = runtime_modules["load_lora_weights"]

        self.model = self._build_model()
        self.transform = self._build_transform()

    def _resolve_repo_path(self, config: dict[str, Any]) -> Path:
        repo_path = resolve_project_relative_path(
            os.environ.get("MEDSAM3_REPO") or config.get("repo_path"),
            SEGMENTATION_ROOT,
            DEFAULT_REPO_PATH,
        )
        if repo_path is None or not repo_path.is_dir():
            raise FileNotFoundError(
                "MedSAM3 repo not found at "
                f"{repo_path}. Clone {self.repo_url} into {DEFAULT_REPO_PATH} "
                "or set MEDSAM3_REPO to an existing checkout."
            )
        return repo_path

    def _read_repo_config(self, config_path: Path) -> dict[str, Any]:
        with config_path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)

    def _resolve_paths(self, config: dict[str, Any]) -> MedSAM3Paths:
        output_dir = self.repo_config.get("output", {}).get("output_dir", "weights/medsam3_v1")
        default_weights_path = self.repo_path / output_dir / "best_lora_weights.pt"
        weights_path = resolve_repo_relative_path(
            config.get("weights_path"),
            self.repo_path,
            default_weights_path,
        )
        if weights_path is None or not weights_path.exists():
            raise FileNotFoundError(
                f"MedSAM3 LoRA weights not found: {weights_path}"
            )

        base_checkpoint_path = resolve_repo_relative_path(
            config.get("base_checkpoint_path"),
            self.repo_path,
        )
        if base_checkpoint_path is not None and not base_checkpoint_path.exists():
            raise FileNotFoundError(
                f"MedSAM3 base checkpoint not found: {base_checkpoint_path}"
            )

        bpe_path = self.repo_path / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"
        if not bpe_path.exists():
            raise FileNotFoundError(f"MedSAM3 tokenizer vocab not found: {bpe_path}")

        return MedSAM3Paths(
            repo_path=self.repo_path,
            repo_config_path=self.repo_config_path,
            weights_path=weights_path,
            base_checkpoint_path=base_checkpoint_path,
            bpe_path=bpe_path,
        )

    def _ensure_repo_on_sys_path(self) -> None:
        repo_path_str = str(self.repo_path)
        if repo_path_str not in sys.path:
            sys.path.insert(0, repo_path_str)

    def _import_runtime_modules(self) -> dict[str, Any]:
        try:
            return {
                "build_sam3_image_model": import_module("sam3.model_builder").build_sam3_image_model,
                "Datapoint": import_module("sam3.train.data.sam3_image_dataset").Datapoint,
                "SAMImage": import_module("sam3.train.data.sam3_image_dataset").Image,
                "FindQueryLoaded": import_module(
                    "sam3.train.data.sam3_image_dataset"
                ).FindQueryLoaded,
                "InferenceMetadata": import_module(
                    "sam3.train.data.sam3_image_dataset"
                ).InferenceMetadata,
                "collate_fn_api": import_module("sam3.train.data.collator").collate_fn_api,
                "copy_data_to_device": import_module("sam3.model.utils.misc").copy_data_to_device,
                "ComposeAPI": import_module("sam3.train.transforms.basic_for_api").ComposeAPI,
                "RandomResizeAPI": import_module(
                    "sam3.train.transforms.basic_for_api"
                ).RandomResizeAPI,
                "ToTensorAPI": import_module("sam3.train.transforms.basic_for_api").ToTensorAPI,
                "NormalizeAPI": import_module(
                    "sam3.train.transforms.basic_for_api"
                ).NormalizeAPI,
                "nms": import_module("torchvision.ops").nms,
                "LoRAConfig": import_module("lora_layers").LoRAConfig,
                "apply_lora_to_model": import_module("lora_layers").apply_lora_to_model,
                "load_lora_weights": import_module("lora_layers").load_lora_weights,
            }
        except ModuleNotFoundError as exc:
            raise ImportError(
                "MedSAM3 dependencies are not available in the current interpreter. "
                "Run the benchmark with the MedSAM3 virtualenv via "
                "scripts/run_medsam_benchmark.sh or install the MedSAM3 "
                "requirements into the active environment."
            ) from exc

    def _build_model(self) -> torch.nn.Module:
        checkpoint_path = None
        should_load_from_hf = self.load_from_hf
        if self.paths.base_checkpoint_path is not None:
            checkpoint_path = str(self.paths.base_checkpoint_path)
            should_load_from_hf = False

        model = self._build_sam3_image_model(
            device=self.device.type,
            compile=False,
            load_from_HF=should_load_from_hf,
            checkpoint_path=checkpoint_path,
            bpe_path=str(self.paths.bpe_path),
            eval_mode=True,
        )

        lora_cfg = self.repo_config["lora"]
        lora_config = self._lora_config_type(
            rank=lora_cfg["rank"],
            alpha=lora_cfg["alpha"],
            dropout=0.0,
            target_modules=lora_cfg["target_modules"],
            apply_to_vision_encoder=lora_cfg["apply_to_vision_encoder"],
            apply_to_text_encoder=lora_cfg["apply_to_text_encoder"],
            apply_to_geometry_encoder=lora_cfg["apply_to_geometry_encoder"],
            apply_to_detr_encoder=lora_cfg["apply_to_detr_encoder"],
            apply_to_detr_decoder=lora_cfg["apply_to_detr_decoder"],
            apply_to_mask_decoder=lora_cfg["apply_to_mask_decoder"],
        )
        model = self._apply_lora_to_model(model, lora_config)
        self._load_lora_weights(model, str(self.paths.weights_path))

        model.to(self.device)
        model.eval()
        return model

    def _build_transform(self) -> Any:
        return self._compose_api(
            transforms=[
                self._random_resize_api(
                    sizes=self.resolution,
                    max_size=self.resolution,
                    square=True,
                    consistent_transform=False,
                ),
                self._to_tensor_api(),
                self._normalize_api(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5],
                ),
            ]
        )

    def _create_datapoint(self, image: Image.Image, text_prompt: str) -> Any:
        width, height = image.size
        return self._datapoint_type(
            find_queries=[
                self._find_query_type(
                    query_text=text_prompt,
                    image_id=0,
                    object_ids_output=[],
                    is_exhaustive=True,
                    query_processing_order=0,
                    inference_metadata=self._inference_metadata_type(
                        coco_image_id=0,
                        original_image_id=0,
                        original_category_id=1,
                        original_size=[width, height],
                        object_id=0,
                        frame_index=0,
                    ),
                )
            ],
            images=[
                self._sam_image_type(
                    data=image,
                    objects=[],
                    size=[height, width],
                )
            ],
        )

    def _extract_scores(self, pred_logits: torch.Tensor) -> torch.Tensor:
        if pred_logits.ndim == 2:
            return pred_logits.sigmoid()[0]
        return pred_logits.sigmoid()[0].max(dim=-1)[0]

    def _scale_boxes_to_image(
        self,
        boxes_cxcywh: torch.Tensor,
        image_size: tuple[int, int],
    ) -> torch.Tensor:
        orig_width, orig_height = image_size
        center_x, center_y, width, height = boxes_cxcywh.unbind(-1)
        x_min = (center_x - width / 2) * orig_width
        y_min = (center_y - height / 2) * orig_height
        x_max = (center_x + width / 2) * orig_width
        y_max = (center_y + height / 2) * orig_height
        return torch.stack([x_min, y_min, x_max, y_max], dim=-1)

    def _resize_masks(
        self,
        masks: torch.Tensor,
        image_size: tuple[int, int],
    ) -> list[np.ndarray]:
        width, height = image_size
        resized = functional.interpolate(
            masks.unsqueeze(0).float(),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        return [
            resized_mask.cpu().numpy().astype(np.uint8)
            for resized_mask in (resized > self.mask_threshold)
        ]

    def _predict_prompt_masks(
        self,
        image: Image.Image,
        text_prompt: str,
    ) -> list[np.ndarray]:
        datapoint = self._create_datapoint(image, text_prompt)
        datapoint = self.transform(datapoint)
        batch = self._collate_fn_api([datapoint], dict_key="input")["input"]
        batch = self._copy_data_to_device(batch, self.device, non_blocking=True)

        outputs = self.model(batch)
        last_output = outputs[-1] if isinstance(outputs, list) else outputs
        pred_masks = last_output.get("pred_masks")
        pred_boxes = last_output.get("pred_boxes")
        pred_logits = last_output.get("pred_logits")
        if pred_masks is None or pred_boxes is None or pred_logits is None:
            return []

        scores = self._extract_scores(pred_logits)
        keep = scores > self.detection_threshold
        if keep.sum().item() == 0:
            return []

        boxes_xyxy = self._scale_boxes_to_image(pred_boxes[0, keep], image.size)
        kept_scores = scores[keep]
        kept_indices = self._nms(boxes_xyxy, kept_scores, self.nms_iou_threshold)
        if kept_indices.numel() == 0:
            return []

        prompt_masks = pred_masks[0, keep][kept_indices].sigmoid()
        return self._resize_masks(prompt_masks, image.size)

    @torch.no_grad()
    def predict(self, image_path: str | Path) -> np.ndarray:
        """Run MedSAM3 text-guided inference and return a binary mask."""
        pil_image = Image.open(image_path).convert("RGB")
        masks = []
        for text_prompt in self.prompts:
            masks.extend(self._predict_prompt_masks(pil_image, text_prompt))

        image_shape = (pil_image.height, pil_image.width)
        return combine_binary_masks(masks, image_shape)
