import pytest
import numpy as np
import math

# Use sys.path modification to import src modules for testing
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.evaluation.metrics import compute_metrics


def test_compute_metrics_perfect_match():
    pred = np.zeros((100, 100))
    gt = np.zeros((100, 100))
    # Create matching 20x20 square
    pred[40:60, 40:60] = 1
    gt[40:60, 40:60] = 1
    
    metrics = compute_metrics(pred, gt)
    assert np.isclose(metrics['dice'], 1.0)
    assert np.isclose(metrics['iou'], 1.0)
    assert np.isclose(metrics['hd95'], 0.0)


def test_compute_metrics_no_overlap():
    pred = np.zeros((100, 100))
    gt = np.zeros((100, 100))
    
    # Non-overlapping squares
    pred[20:40, 20:40] = 1
    gt[60:80, 60:80] = 1
    
    metrics = compute_metrics(pred, gt)
    assert np.isclose(metrics['dice'], 0.0)
    assert np.isclose(metrics['iou'], 0.0)
    assert metrics['hd95'] > 10.0  # HD95 should be large


def test_compute_metrics_half_overlap():
    pred = np.zeros((100, 100))
    gt = np.zeros((100, 100))
    
    # 20x20 square vs 20x40 rectangle sharing 20x20
    pred[40:60, 40:60] = 1
    gt[40:60, 40:80] = 1
    
    metrics = compute_metrics(pred, gt)
    
    # pred area = 400
    # gt area = 800
    # intersection = 400
    # dice = 2 * 400 / (400 + 800) = 800 / 1200 = 2/3 ≈ 0.666
    # iou = 400 / (400 + 800 - 400) = 400 / 800 = 1/2 = 0.5
    assert np.isclose(metrics['dice'], 2/3)
    assert np.isclose(metrics['iou'], 0.5)
    assert metrics['hd95'] > 0.0


def test_compute_metrics_both_empty():
    pred = np.zeros((100, 100))
    gt = np.zeros((100, 100))
    
    metrics = compute_metrics(pred, gt)
    assert np.isclose(metrics['dice'], 1.0)
    assert np.isclose(metrics['iou'], 1.0)
    assert math.isnan(metrics['hd95'])


def test_compute_metrics_pred_empty():
    pred = np.zeros((100, 100))
    gt = np.zeros((100, 100))
    gt[40:60, 40:60] = 1
    
    metrics = compute_metrics(pred, gt)
    assert np.isclose(metrics['dice'], 0.0)
    assert np.isclose(metrics['iou'], 0.0)
    assert math.isnan(metrics['hd95'])


def test_compute_metrics_gt_empty():
    pred = np.zeros((100, 100))
    gt = np.zeros((100, 100))
    pred[40:60, 40:60] = 1
    
    metrics = compute_metrics(pred, gt)
    assert np.isclose(metrics['dice'], 0.0)
    assert np.isclose(metrics['iou'], 0.0)
    assert math.isnan(metrics['hd95'])


def test_compute_metrics_shape_mismatch():
    pred = np.zeros((100, 100))
    gt = np.zeros((50, 50))
    
    with pytest.raises(ValueError, match="Shape mismatch"):
        compute_metrics(pred, gt)
