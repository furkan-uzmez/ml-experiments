$ErrorActionPreference = "Stop"

Write-Host "==========================================================="
Write-Host "   U-Net Baseline Benchmark Runner - ISIC 2018 Task 1     "
Write-Host "==========================================================="

# Add src to PYTHONPATH
$env:PYTHONPATH = "$PWD;$env:PYTHONPATH"

# Run configurations
$SEEDS = @(11, 22, 33)

foreach ($SEED in $SEEDS) {
    Write-Host ""
    Write-Host ">>> Starting Pipeline for SEED: $SEED <<<"
    Write-Host "-----------------------------------------------------------"
    
    Write-Host "[1/3] Training U-Net ($SEED)..."
    python src/train/train_unet.py --seed $SEED
    
    Write-Host "[2/3] Running Inference ($SEED)..."
    python src/infer/infer_unet.py --seed $SEED
    
    Write-Host "[3/3] Validation & Extraction Done for $SEED."
    Write-Host "-----------------------------------------------------------"
}

Write-Host ""
Write-Host "==========================================================="
Write-Host "   Benchmark Completed for all seeds!                     "
Write-Host "   Checkpoints in: artifacts/unet/checkpoints/            "
Write-Host "   Predictions in: artifacts/unet/predictions/            "
Write-Host "==========================================================="
