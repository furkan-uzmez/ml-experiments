#!/bin/bash
set -e

# Setup environment if needed (assuming user has conda env 'ml' active)
echo "==========================================================="
echo "   U-Net Baseline Benchmark Runner - ISIC 2018 Task 1     "
echo "==========================================================="

# Add src to PYTHONPATH
export PYTHONPATH="$(pwd):$PYTHONPATH"

# Run configurations
SEEDS=(11 22 33)

for SEED in "${SEEDS[@]}"
do
    echo ""
    echo ">>> Starting Pipeline for SEED: $SEED <<<"
    echo "-----------------------------------------------------------"
    
    echo "[1/3] Training U-Net ($SEED)..."
    python3 src/train/train_unet.py --seed $SEED
    
    echo "[2/3] Running Inference ($SEED)..."
    python3 src/infer/infer_unet.py --seed $SEED
    
    echo "[3/3] Validation & Extraction Done for $SEED."
    echo "-----------------------------------------------------------"
done

echo ""
echo "==========================================================="
echo "   Benchmark Completed for all seeds!                     "
echo "   Checkpoints in: artifacts/unet/checkpoints/            "
echo "   Predictions in: artifacts/unet/predictions/            "
echo "==========================================================="
