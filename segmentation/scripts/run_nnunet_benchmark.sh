#!/bin/bash
set -e

echo "==========================================================="
echo "   nnU-Net Benchmark Runner - ISIC 2018 Task 1     "
echo "==========================================================="

export PYTHONPATH="$(pwd):$PYTHONPATH"

# nnU-Net specific environment variables
export nnUNet_raw="$(pwd)/artifacts/nnunet/nnUNet_raw"
export nnUNet_preprocessed="$(pwd)/artifacts/nnunet/nnUNet_preprocessed"
export nnUNet_results="$(pwd)/artifacts/nnunet/nnUNet_results"

DATASET_ID="500"
DATASET_NAME="Dataset500_ISIC2018"
TEST_FOLDER="$nnUNet_raw/$DATASET_NAME/imagesTs"

SEEDS=(11 22 33)

echo ">>> Phase 1: Validating Directory Structure <<<"
mkdir -p "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results"

# 1. Convert Data to nnU-Net standard if not existing
if [ ! -d "$nnUNet_raw/$DATASET_NAME" ]; then
    echo ">>> Phase 2: Converting ISIC to nnUNet Format (Modality Splitting) <<<"
    python3 src/dataio/convert_to_nnunet.py
else
    echo ">>> Phase 2: Found existing formatted dataset, skipping conversion. <<<"
fi

# 2. Plan and Preprocess
if [ ! -d "$nnUNet_preprocessed/$DATASET_NAME" ]; then
    echo ">>> Phase 3: nnUNet Plan and Preprocess (2D) <<<"
    # We use -c 2d because our config mandates a strict 2D evaluation space
    nnUNetv2_plan_and_preprocess -d $DATASET_ID --verify_dataset_integrity -c 2d
else
    echo ">>> Phase 3: Found existing preprocessed dataset, skipping plan & preprocess. <<<"
fi

# 3. Apply the predefined Benchmark Cross-Validation Split (Fold 0 ONLY)
echo ">>> Phase 4: Enforcing Benchmark Deterministic Split <<<"
python3 src/dataio/enforce_nnunet_split.py


# 4. Train and Infer for Each Seed
for SEED in "${SEEDS[@]}"
do
    echo ""
    echo ">>> Starting Pipeline for SEED: $SEED <<<"
    echo "-----------------------------------------------------------"
    
    # Python environment determinism passed to underlying script processes
    export PYTHONHASHSEED=$SEED
    export CUBLAS_WORKSPACE_CONFIG=:4096:8
    
    echo "[1/3] Training nnU-Net (Fold 0, Seed $SEED)..."
    # Using random seed directly to limit variability. Fold 0 maps to our strict `primary_split.json`.
    # Run in background? We will run synchronously so the user can see.
    nnUNetv2_train $DATASET_ID 2d 0 -tr nnUNetTrainer_250epochs
    
    PRED_DIR="$nnUNet_results/$DATASET_NAME/predictions_seed_$SEED"
    mkdir -p "$PRED_DIR"
    
    echo "[2/3] Running Inference on Test Set (Seed $SEED)..."
    # Note: the test images were routed to `imagesTs` during the conversion step.
    nnUNetv2_predict -i "$TEST_FOLDER" -o "$PRED_DIR" -d $DATASET_ID -c 2d -f 0
    
    echo "[3/3] Validating and Computing Metrics..."
    python3 src/infer/eval_nnunet.py --pred_dir "$PRED_DIR" --seed $SEED
    
    echo "-----------------------------------------------------------"
done

echo ""
echo "==========================================================="
echo "   Benchmark Completed for all seeds!                     "
echo "   Aggregated JSON Metrics in: $nnUNet_results/$DATASET_NAME/     "
echo "==========================================================="
