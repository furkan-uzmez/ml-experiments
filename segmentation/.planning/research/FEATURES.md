# Features Research — Medical Image Segmentation Benchmark

## Table Stakes (Must Have — Users/Reviewers Expect These)

### Data Management
- **Patient-level data splitting** — Prevents data leakage; mandatory for valid evaluation
- **Deterministic split files** — JSON/CSV with patient IDs, reproducible across runs
- **Consistent class definitions** — Identical label mapping for all three models

### Evaluation Protocol
- **Dice Similarity Coefficient (DSC)** — Primary overlap metric, universally expected
- **Intersection over Union (IoU/Jaccard)** — Complementary overlap metric
- **95th Percentile Hausdorff Distance (HD95)** — Boundary quality metric, critical for clinical relevance
- **Per-case metric reporting** — Individual case results, not just aggregates
- **Mean ± standard deviation** — Summary statistics across test cases and runs
- **Common evaluation space** — All predictions resampled to identical resolution before metric computation

### Reproducibility
- **Fixed random seeds** — Same seeds (e.g., [11, 22, 33]) for all stochastic elements
- **Multiple training runs** — At minimum 3 runs for variance estimation
- **Version pinning** — Exact library versions recorded
- **Configuration files** — All hyperparameters in YAML/JSON, not hardcoded

### Operational Metrics
- **Inference time per case** — Wall-clock time for single-case prediction
- **Peak GPU memory** — Maximum VRAM usage during inference

## Differentiators (Competitive Advantage for a Strong Paper)

- **Normalized Surface Dice (NSD)** — More nuanced boundary metric at specific tolerance thresholds
- **Per-class breakdown** — Metrics for each segmentation class separately
- **Statistical significance testing** — Wilcoxon signed-rank tests between models
- **Failure case analysis** — Qualitative examination of worst-performing cases
- **Computational cost analysis** — FLOPs, training time, parameter counts
- **Effect size reporting** — Cohen's d or similar for model comparisons
- **Ablation on dataset size** — How models degrade with less training data

## Anti-Features (Deliberately NOT Building)

- **Ensemble of different models** — Benchmark goal is individual model comparison, not ensemble performance
- **Architecture search or modification** — Use models as provided; nnU-Net in particular should not be architecturally modified
- **Multi-dataset evaluation** — Explicitly single-dataset benchmark for controlled comparison
- **Real-time deployment pipeline** — Focus on accuracy and reproducibility, not deployment
- **Interactive segmentation interface** — Benchmark is batch-mode, not interactive

## Feature Dependencies

```
Data Split → All Model Training
           → All Model Inference
           → Shared Evaluation

Shared Evaluation ← U-Net Predictions
                  ← nnU-Net Predictions (with adapter)
                  ← MedSAM Predictions

Results Aggregation ← Per-case Metrics
                   → Comparison Table
                   → Figures
                   → Report
```
