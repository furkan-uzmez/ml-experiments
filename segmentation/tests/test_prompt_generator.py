import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.infer.prompt_generator import PromptGenerator

def test_generate_bbox_normal():
    # 100x100 mask
    gt_mask = np.zeros((100, 100))
    # Fill from r=40..60, c=40..60
    gt_mask[40:60, 40:60] = 1
    
    gen = PromptGenerator(padding=5, strategy="macro")
    bbox = gen.generate_bbox(gt_mask)
    
    # x_min = min(col) = 40. pad = 5 -> 35
    # y_min = min(row) = 40. pad = 5 -> 35
    # x_max = max(col) = 59. pad = 5 -> 64
    # y_max = max(row) = 59. pad = 5 -> 64
    # bbox should be [35, 35, 64, 64]
    
    assert bbox is not None
    assert np.array_equal(bbox, np.array([35, 35, 64, 64]))

def test_generate_bbox_padding_boundary():
    # Top left corner
    gt_mask = np.zeros((100, 100))
    gt_mask[0:10, 0:10] = 1
    
    gen = PromptGenerator(padding=15, strategy="macro")
    bbox = gen.generate_bbox(gt_mask)
    
    # x_min = 0 - 15 -> clamp to 0
    # y_min = 0 - 15 -> clamp to 0
    # x_max = 9 + 15 = 24
    # y_max = 9 + 15 = 24
    
    assert bbox is not None
    assert np.array_equal(bbox, np.array([0, 0, 24, 24]))

def test_generate_bbox_empty():
    gt_mask = np.zeros((100, 100))
    gen = PromptGenerator(padding=5, strategy="macro")
    bbox = gen.generate_bbox(gt_mask)
    assert bbox is None
