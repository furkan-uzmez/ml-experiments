import os
import pandas as pd
from datetime import datetime

def generate_markdown_report(csv_path: str, out_dir: str):
    """Compiles the final markdown report comparing the benchmark runs."""
    if not os.path.exists(csv_path):
        print("Missing aggregate CSV.")
        return
        
    df = pd.read_csv(csv_path)
    if df.empty:
        return
        
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Compute statistical aggregations
    # Average across seeds first for each case
    df_avg = df.groupby(["model", "case_id"], as_index=False).mean(numeric_only=True)
    # Then overall mean/std per model
    stats = df_avg.groupby("model")[["dice", "iou", "hd95"]].agg(['mean', 'std'])
    
    # Restructure for easy formatting
    results_table = "| Model | Dice (Mean ± Std) | IoU (Mean ± Std) | HD95 |\\n|-------|-------------------|------------------|------|\\n"
    
    best_model = None
    best_dice = -1.0
    
    for model in stats.index:
        d_mean = stats.loc[model, ('dice', 'mean')]
        d_std = stats.loc[model, ('dice', 'std')]
        i_mean = stats.loc[model, ('iou', 'mean')]
        i_std = stats.loc[model, ('iou', 'std')]
        h_mean = stats.loc[model, ('hd95', 'mean')]
        
        if d_mean > best_dice:
            best_dice = d_mean
            best_model = model
            
        results_table += f"| {model} | {d_mean:.4f} ± {d_std:.4f} | {i_mean:.4f} ± {i_std:.4f} | {h_mean:.2f} |\\n"
        
    # Markdown Template
    report = f"""# ISIC 2018 Segmentation Benchmark Report

**Generated On**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

This report outlines the comparative evaluation of the U-Net baseline, the self-configuring nnU-Net architecture, and the prompt-conditioned foundation model (MedSAM).

## 1. System Protocol
- **Dataset**: ISIC 2018 Task 1 (Dermoscopy)
- **Evaluation Split**: Deterministic Hold-out Test split.
- **Metrics**: Native-resolution (resampling disabled models back to original mask space).

## 2. Quantitative Results

{results_table}

**Winner**: The architecture achieving the highest average Dice coefficient across all evaluated test cases is **{best_model}** ({best_dice:.4f}).

## 3. Visual Comparisons

Below are the aggregated statistical visualizations showcasing the robustness and variance of each model across the test cohort.

### Dice Similarity Coefficient (Higher is Better)
![Dice Boxplot](figures/dice_boxplot.png)

### Intersection over Union (Higher is Better)
![IoU Boxplot](figures/iou_boxplot.png)

### Hausdorff Distance 95% (Lower is Better)
*(Note: Extreme outliers >99th percentile are clipped for plotting legibility)*
![HD95 Boxplot](figures/hd95_boxplot.png)

---
*Generated automatically by Phase 6 Pipeline.*
"""

    out_file = os.path.join(out_dir, "BENCHMARK_REPORT.md")
    with open(out_file, "w") as f:
        f.write(report.replace('\\n', '\n'))
        
    print(f"Report generated successfully -> {out_file}")

if __name__ == "__main__":
    generate_markdown_report("reports/benchmark_results.csv", "reports")
