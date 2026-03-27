# Roadmap — Medical Image Segmentation Benchmark

> ISIC 2018 Task 1: U-Net vs nnU-Net vs MedSAM

## Current Milestone: v1.0 — Single-Dataset Benchmark

**Status**: Planning
**Phases**: 6
**Requirements**: 35

---

## Phase 1: Benchmark Contract & Data Split

**Goal**: Define the benchmark contract, fixed data split, and reproducibility rules before any training starts.

**Requirements**: DATA-01, DATA-02, DATA-03, DATA-04, DATA-05, DATA-06, REPRO-01, REPRO-04, REPRO-05

**UI hint**: no

**Success Criteria**:
1. `configs/dataset.yaml` exists with ISIC 2018 Task 1 contract (path, modality=dermoscopy, labels=binary, dimensionality=2D)
2. `splits/primary_split.json` exists with patient-level train/val/test split and zero overlap verified
3. `configs/seeds.yaml` exists with [11, 22, 33]
4. `configs/metrics.yaml` exists defining Dice, IoU, HD95, latency, peak memory
5. `docs/benchmark_protocol.md` exists documenting fairness rules and MedSAM prompt protocol

---

## Phase 2: Evaluation Scaffold

**Goal**: Build the shared data loading and evaluation pipeline used by all three model pipelines.

**Requirements**: EVAL-01, EVAL-02, EVAL-03, EVAL-04, EVAL-05, EVAL-06, EVAL-07, EVAL-08, DATA-04, DATA-05, REPRO-02

**UI hint**: no

**Success Criteria**:
1. `src/dataio/dataset_index.py` loads ISIC 2018 image-mask pairs mapped to patient IDs
2. `src/dataio/split_loader.py` enforces primary split across all pipelines
3. `src/evaluation/metrics.py` computes Dice, IoU, HD95 with correct edge-case handling (validated on synthetic masks)
4. `src/evaluation/runtime.py` records inference time and peak GPU memory
5. `src/evaluation/resample.py` transforms predictions to common evaluation space (2D resize to matching resolution)

---

## Phase 3: U-Net Baseline Pipeline

**Goal**: Implement the U-Net benchmark pipeline as the simplest supervised baseline and validate the full train→infer→eval flow.

**Requirements**: UNET-01, UNET-02, UNET-03, UNET-04, UNET-05, REPRO-01, REPRO-03

**UI hint**: no

**Success Criteria**:
1. `configs/unet.yaml` defines all hyperparameters (input shape, batch size, optimizer, LR, loss, epochs, augmentation)
2. `src/models/unet_model.py` implements a 2D U-Net appropriate for ISIC 2018
3. `src/train/train_unet.py` completes 3 training runs with seeds [11, 22, 33]
4. `src/infer/infer_unet.py` saves predictions to `artifacts/unet/predictions/<seed>/`
5. `scripts/run_unet_benchmark.sh` executes full pipeline and produces evaluation metrics

---

## Phase 4: nnU-Net Integration

**Goal**: Integrate nnU-Net as the automated strong baseline while preserving the benchmark contract.

**Requirements**: NNUNET-01, NNUNET-02, NNUNET-03, NNUNET-04, NNUNET-05, NNUNET-06

**UI hint**: no

**Success Criteria**:
1. `src/adapters/nnunet_dataset_conversion.py` converts ISIC 2018 to nnU-Net format maintaining split semantics
2. nnU-Net 2D trainer runs with automatic configuration (no manual architecture changes)
3. nnU-Net uses benchmark's primary split (not default 5-fold)
4. `src/adapters/nnunet_prediction_import.py` maps nnU-Net outputs to common evaluation space
5. `artifacts/nnunet/run_manifest.json` records version, planner, and trainer settings

---

## Phase 5: MedSAM Integration

**Goal**: Integrate MedSAM with deterministic bounding-box prompt protocol for ISIC 2018.

**Requirements**: MEDSAM-01, MEDSAM-02, MEDSAM-03, MEDSAM-04, MEDSAM-05

**UI hint**: no

**Success Criteria**:
1. `configs/medsam.yaml` declares fixed mode (fine-tuned or zero-shot) and prompt generation rules
2. `src/prompts/medsam_prompting.py` generates deterministic bounding-box prompts from ground-truth masks
3. Training/fine-tuning uses same split and seeds [11, 22, 33] (if fine-tuned)
4. Predictions saved to `artifacts/medsam/predictions/<seed_or_mode>/`
5. Prompt protocol documented in `docs/benchmark_protocol.md`

---

## Phase 6: Results Aggregation & Report

**Goal**: Aggregate results and produce the final benchmark comparison package.

**Requirements**: REPORT-01, REPORT-02, REPORT-03, REPORT-04, REPORT-05

**UI hint**: no

**Success Criteria**:
1. `artifacts/summary/model_comparison.csv` contains one row per model-run and aggregated per-model
2. Boxplots and bar charts generated under `artifacts/summary/figures/`
3. `reports/benchmark_report.md` includes dataset definition, protocol, quantitative results, qualitative failure cases, recommendation
4. `artifacts/summary/run_registry.json` documents all runs with model, seed, commit hash, config, artifact paths
5. `scripts/run_full_benchmark.sh` orchestrates all three pipelines and report generation
