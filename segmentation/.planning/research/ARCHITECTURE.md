# Architecture Research — Medical Image Segmentation Benchmark

## System Components

```
┌─────────────────────────────────────────────────────────────┐
│                    BENCHMARK SYSTEM                         │
├──────────┬──────────┬──────────┬──────────┬─────────────────┤
│  Config  │  Data    │  Models  │  Eval    │  Reporting      │
│  Layer   │  Layer   │  Layer   │  Layer   │  Layer          │
├──────────┼──────────┼──────────┼──────────┼─────────────────┤
│ dataset  │ index    │ U-Net    │ resample │ aggregate       │
│ seeds    │ split    │ nnU-Net  │ metrics  │ plots           │
│ metrics  │ loader   │ MedSAM   │ runtime  │ report          │
│ model    │          │          │          │                 │
│ configs  │          │          │          │                 │
└──────────┴──────────┴──────────┴──────────┴─────────────────┘
```

## Component Boundaries

### 1. Configuration Layer (`configs/`)
- **Input**: User-defined YAML files
- **Output**: Structured config objects consumed by all other layers
- **Boundary**: Read-only during benchmark execution; configs are immutable after setup
- **Files**: `dataset.yaml`, `seeds.yaml`, `metrics.yaml`, `unet.yaml`, `nnunet.yaml`, `medsam.yaml`

### 2. Data Layer (`src/dataio/`)
- **Input**: Raw dataset, config, split file
- **Output**: Indexed patient-image-label pairs with split assignments
- **Boundary**: Does NOT perform model-specific preprocessing; just loads and indexes
- **Key Constraint**: nnU-Net has its own data format; this layer provides the standard split that nnU-Net's adapter maps into

### 3. Model Layer (`src/models/`, `src/train/`, `src/infer/`, `src/adapters/`, `src/prompts/`)
- **Input**: Data from Data Layer, configs
- **Output**: Predictions saved to `artifacts/<model>/predictions/<seed>/`
- **Boundary**: Each model has its own train/infer pipeline; they share ONLY the data split and configuration
- **Key Design**: nnU-Net and MedSAM use adapters to bridge between the benchmark's data format and model-native formats

### 4. Evaluation Layer (`src/evaluation/`)
- **Input**: Predictions from all models, ground truth from Data Layer
- **Output**: Per-case metrics, runtime measurements
- **Boundary**: Identical evaluation code for all models; predictions are resampled to common space first
- **Critical Rule**: Evaluation is model-agnostic — it only sees (prediction, ground_truth) pairs

### 5. Reporting Layer (`src/reporting/`)
- **Input**: Metric outputs from Evaluation Layer
- **Output**: CSV tables, figures, markdown report
- **Boundary**: Pure aggregation and visualization; no computation of new metrics

## Data Flow

```
configs/dataset.yaml ──→ dataset_index.py ──→ patient_id_list
                                                    │
splits/primary_split.json ──→ split_loader.py ──→ train/val/test assignments
                                                    │
                                    ┌───────────────┼───────────────┐
                                    ▼               ▼               ▼
                              train_unet.py   nnunet_adapter    train_medsam.py
                                    │               │               │
                              infer_unet.py   nnunet_infer      infer_medsam.py
                                    │               │               │
                              predictions/    nnunet_import     predictions/
                              unet/<seed>/    predictions/      medsam/<seed>/
                                    │         nnunet/               │
                                    └───────────┼───────────────────┘
                                                ▼
                                         resample.py (common eval space)
                                                ▼
                                         metrics.py (Dice, IoU, HD95)
                                         runtime.py (latency, GPU mem)
                                                ▼
                                     aggregate_results.py
                                                ▼
                                  model_comparison.csv + figures/
                                                ▼
                                      benchmark_report.md
```

## Suggested Build Order

| Order | Component | Rationale |
|-------|-----------|-----------|
| 1 | Config Layer | Foundation; all components depend on configs |
| 2 | Data Layer (index + split) | Required before any training |
| 3 | Evaluation Layer | Can be tested with synthetic data; needed by all models |
| 4 | U-Net Pipeline | Simplest model; validates the full train→infer→eval flow |
| 5 | nnU-Net Pipeline | Most complex adapter logic; builds on validated eval pipeline |
| 6 | MedSAM Pipeline | Prompt generation adds complexity; builds on validated eval pipeline |
| 7 | Reporting Layer | Aggregates all results; built last |

## Key Architecture Decisions

- **Adapter pattern for nnU-Net**: Rather than modifying nnU-Net, adapters translate between benchmark and nnU-Net formats
- **Shared evaluation, separate training**: Training pipelines are independent; evaluation is universal
- **Prediction-centric storage**: All models save predictions to a standard format in `artifacts/`; evaluation reads from there
- **Immutable configs and splits**: Once set, configs and splits never change during a benchmark run
