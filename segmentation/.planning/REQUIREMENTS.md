# Requirements — Medical Image Segmentation Benchmark

> v1 Benchmark: U-Net vs nnU-Net vs MedSAM on a single dataset

## v1 Requirements

### Data & Split

- [x] **DATA-01**: Define dataset contract (root path, modality, target labels, spacing, normalization, dimensionality) in `configs/dataset.yaml`
- [x] **DATA-02**: Create deterministic patient-level train/validation/test split in `splits/primary_split.json`
- [x] **DATA-03**: Verify zero patient overlap across train/val/test sets (automated test)
- [x] **DATA-04**: Index all image-label pairs mapped to patient identifiers
- [x] **DATA-05**: Enforce same split across all model pipelines
- [x] **DATA-06**: Remove or avoid direct patient identifiers in benchmark artifacts

### Evaluation

- [ ] **EVAL-01**: Compute Dice Similarity Coefficient (DSC) per case and aggregated (mean ± std)
- [ ] **EVAL-02**: Compute Intersection over Union (IoU/Jaccard) per case and aggregated
- [ ] **EVAL-03**: Compute 95th Percentile Hausdorff Distance (HD95) per case and aggregated
- [ ] **EVAL-04**: Record inference time per case (wall-clock)
- [ ] **EVAL-05**: Record peak GPU memory during inference
- [ ] **EVAL-06**: Resample all predictions and ground truth to one common evaluation space before metric computation
- [ ] **EVAL-07**: Use identical metric implementation for all three models
- [ ] **EVAL-08**: Handle edge cases (empty predictions, empty ground truth) explicitly and consistently

### U-Net Pipeline

- [ ] **UNET-01**: Define U-Net configuration (input shape, batch size, optimizer, LR, loss, epochs, augmentation, checkpoint policy)
- [ ] **UNET-02**: Implement baseline U-Net appropriate for dataset dimensionality
- [ ] **UNET-03**: Train with 3 repeated runs using seeds [11, 22, 33]
- [ ] **UNET-04**: Save full-resolution predictions to `artifacts/unet/predictions/<seed>/`
- [ ] **UNET-05**: Execute train → inference → evaluation in fixed order

### nnU-Net Pipeline

- [ ] **NNUNET-01**: Document dataset conversion path, trainer command, folds strategy, inference command
- [ ] **NNUNET-02**: Convert primary dataset to nnU-Net input format preserving split semantics
- [ ] **NNUNET-03**: Run nnU-Net with prescribed automatic configuration (no manual architecture changes)
- [ ] **NNUNET-04**: Use benchmark's primary split (not nnU-Net default 5-fold)
- [ ] **NNUNET-05**: Map nnU-Net outputs into common evaluation space
- [ ] **NNUNET-06**: Record exact nnU-Net version, planner, and trainer settings in run manifest

### MedSAM Pipeline

- [ ] **MEDSAM-01**: Declare fixed mode (fine-tuned OR zero-shot) consistently
- [ ] **MEDSAM-02**: Define deterministic prompt generation rules (prompt type, source, image resizing)
- [ ] **MEDSAM-03**: Train/fine-tune using same split and seeds [11, 22, 33] (if fine-tuned mode)
- [ ] **MEDSAM-04**: Save predictions to `artifacts/medsam/predictions/<seed_or_mode>/`
- [ ] **MEDSAM-05**: Document prompt protocol in benchmark protocol document

### Reproducibility

- [x] **REPRO-01**: Fix random seeds across PyTorch, NumPy, Python random, CUDA
- [ ] **REPRO-02**: Use same class definitions and label mapping for all three models
- [ ] **REPRO-03**: Store predictions for complete test set for all models
- [x] **REPRO-04**: Store configuration files externally (not hardcoded)
- [x] **REPRO-05**: Document benchmark protocol (fairness rules, execution policy, MedSAM prompt protocol)

### Reporting

- [ ] **REPORT-01**: Aggregate all run outputs into `artifacts/summary/model_comparison.csv`
- [ ] **REPORT-02**: Produce boxplots and per-metric bar charts
- [ ] **REPORT-03**: Create benchmark report with dataset definition, protocol, quantitative results, qualitative failure cases, recommendation
- [ ] **REPORT-04**: Create run registry listing model name, seed, commit hash, config path, artifact paths
- [ ] **REPORT-05**: Orchestrate all three pipelines and report generation from a single script

## v2 Requirements (Deferred)

- Statistical significance testing (Wilcoxon signed-rank between models)
- Normalized Surface Dice (NSD) at specific tolerance thresholds
- Per-class metric breakdown
- Failure case visualization (worst-performing cases with overlays)
- Ablation on dataset size (training with 25%, 50%, 75%, 100% data)
- Computational cost analysis (FLOPs, parameter counts, training time)
- Confidence intervals via bootstrap sampling

## Out of Scope

- **Multi-dataset evaluation** — Benchmark is explicitly single-dataset for controlled comparison
- **Ensemble of models** — Goal is individual model comparison
- **Architecture modifications to nnU-Net** — Must run with automatic configuration
- **Real-time deployment** — Focus on accuracy and reproducibility
- **Interactive segmentation** — Batch-mode benchmark only
- **Custom U-Net variants** — Only standard baseline U-Net

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| DATA-01..06 | Phase 1: Benchmark Contract & Data Split | ✅ Done |
| EVAL-01..08 | Phase 2: Evaluation Scaffold | ✅ Done |
| UNET-01..05 | Phase 3: U-Net Baseline Pipeline | ✅ Done |
| NNUNET-01..06 | Phase 4: nnU-Net Integration | ✅ Done |
| MEDSAM-01..05 | Phase 5: MedSAM Integration | ✅ Done |
| REPRO-01,04,05 | Phase 1: Benchmark Contract & Data Split | ✅ Done |
| REPRO-02,03 | Phase 2+ (cross-cutting) | Pending |
| REPORT-01..05 | Phase 6: Results Aggregation & Report | Pending |
