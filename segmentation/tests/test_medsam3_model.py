import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.medsam3_model import (
    combine_binary_masks,
    normalize_text_prompts,
    resolve_project_relative_path,
    resolve_repo_relative_path,
)


def test_normalize_text_prompts_keeps_non_empty_values():
    prompts = normalize_text_prompts([" skin lesion ", "", "lesion"])
    assert prompts == ["skin lesion", "lesion"]


def test_normalize_text_prompts_uses_fallback_for_empty_inputs():
    prompts = normalize_text_prompts([], fallback_prompt="lesion")
    assert prompts == ["lesion"]


def test_combine_binary_masks_merges_foreground_pixels():
    first_mask = np.array([[0, 1], [0, 0]], dtype=np.uint8)
    second_mask = np.array([[0, 0], [1, 0]], dtype=np.uint8)

    combined = combine_binary_masks([first_mask, second_mask], image_shape=(2, 2))

    assert np.array_equal(combined, np.array([[0, 1], [1, 0]], dtype=np.uint8))


def test_combine_binary_masks_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        combine_binary_masks([np.zeros((3, 3), dtype=np.uint8)], image_shape=(2, 2))


def test_resolve_repo_relative_path_anchors_to_repo_root():
    repo_path = Path("/tmp/medsam3")
    resolved = resolve_repo_relative_path("weights/model.pt", repo_path)
    assert resolved == (repo_path / "weights" / "model.pt").resolve()


def test_resolve_project_relative_path_anchors_to_segmentation_root():
    project_root = Path("/tmp/segmentation")
    resolved = resolve_project_relative_path("external/MedSAM3", project_root)
    assert resolved == (project_root / "external" / "MedSAM3").resolve()
