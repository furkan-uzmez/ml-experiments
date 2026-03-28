#!/bin/bash
set -e

echo "==========================================================="
echo "   MedSAM Benchmark Evaluator - ISIC 2018 Task 1     "
echo "==========================================================="

export PYTHONPATH="$(pwd):$PYTHONPATH"

# MedSAM has no "training" loop for our benchmark (Zero-Shot execution).
# The only requirement is that MedSAM's ViT-b weights are accessible.

echo ">>> Phase 1: Validating Environment & Configurations <<<"
# Configs parse automatically within the infer script.

echo ">>> Phase 2: Running Deterministic Zero-Shot Inference <<<"
# MedSAM uses a pre-trained network. No SEED loop is fully necessary 
# because BBOX extraction and MedSAM forward passes are intrinsically deterministic.
# We run once and save the predictions to the unified artifacts format.

python3 src/infer/infer_medsam.py

echo "==========================================================="
echo "   MedSAM Benchmark Completed!                            "
echo "   Aggregated JSON Metrics in: artifacts/medsam/predictions/ "
echo "==========================================================="
