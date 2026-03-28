#!/bin/bash
set -e

echo "==========================================================="
echo "   Benchmark Reporting Pipeline - Phase 6                  "
echo "==========================================================="

export PYTHONPATH="$(pwd):$PYTHONPATH"

echo ">>> Phase 1/3: Reading and Aggregating Test Set Metrics <<<"
# This evaluates all generated predictions on the fly and saves the unified dataset.
# The advantage is metrics are computed identically one final time regardless of caching.
python3 src/reporting/aggregate_metrics.py

echo ">>> Phase 2/3: Generating Matplotlib/Seaborn Visualizations <<<"
python3 src/reporting/visualize.py

echo ">>> Phase 3/3: Assembling Markdown Report <<<"
python3 src/reporting/generate_report.py

echo "==========================================================="
echo "   Reporting Complete!                                    "
echo "   Outputs located in: reports/                           "
echo "==========================================================="
