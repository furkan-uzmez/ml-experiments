#!/bin/bash
set -e

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SEGMENTATION_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
DEFAULT_MEDSAM3_REPO="$SEGMENTATION_ROOT/external/MedSAM3"
MEDSAM3_REPO_URL="https://github.com/Joey-S-Liu/MedSAM3"

echo "==========================================================="
echo "   MedSAM3 Benchmark Evaluator - ISIC 2018 Task 1    "
echo "==========================================================="

MEDSAM3_REPO="${MEDSAM3_REPO:-$DEFAULT_MEDSAM3_REPO}"
MEDSAM3_PYTHON_BIN="${MEDSAM3_PYTHON_BIN:-$MEDSAM3_REPO/.venv/bin/python}"

if [ ! -d "$MEDSAM3_REPO" ]; then
  echo "MedSAM3 repo not found: $MEDSAM3_REPO"
  echo "Clone the official repo with:"
  echo "  git clone $MEDSAM3_REPO_URL \"$MEDSAM3_REPO\""
  exit 1
fi

if [ ! -x "$MEDSAM3_PYTHON_BIN" ]; then
  MEDSAM3_PYTHON_BIN="python3"
fi

export PYTHONPATH="$SEGMENTATION_ROOT:$MEDSAM3_REPO:$PYTHONPATH"

cd "$SEGMENTATION_ROOT"

# MedSAM3 runs as a pre-trained text-guided model in this benchmark.
# The runtime comes from the official MedSAM3 checkout inside this workspace,
# or from a path supplied via MEDSAM3_REPO.

echo ">>> Phase 1: Validating Environment & Configurations <<<"

echo ">>> Phase 2: Running Deterministic Zero-Shot Inference <<<"

"$MEDSAM3_PYTHON_BIN" src/infer/infer_medsam.py

echo "==========================================================="
echo "   MedSAM3 Benchmark Completed!                           "
echo "   Aggregated JSON Metrics in: artifacts/medsam3/predictions/ "
echo "==========================================================="
