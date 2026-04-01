# Benchmark Protocol — Medical Image Segmentation

## Dataset

- **Name**: ISIC 2018 Task 1 (Lesion Boundary Segmentation)
- **Modality**: Dermoscopy (2D RGB)
- **Task**: Binary segmentation (lesion vs background)
- **Total Samples**: 2594 images with ground truth masks
- **Image Format**: JPEG (RGB) — variable sizes
- **Mask Format**: PNG (grayscale, binary) — matching sizes

## Split Policy

- **Method**: Patient-level random split with fixed seed (seed=42)
- **Ratios**: 70% train / 15% validation / 15% test
- **Counts**: 1815 train / 389 validation / 390 test
- **Split File**: `splits/primary_split.json`
- **Immutability**: Split is fixed for all models and all runs — never modified after creation
- **Note**: Each ISIC image represents a unique patient (one image per patient)

## Fairness Rules

### 1. Identical Data Access
All three models (U-Net, nnU-Net, MedSAM) use the exact same train/val/test patient assignments from `primary_split.json`.

### 2. Independent Preprocessing
- **U-Net**: Standardized resize + normalization defined in `configs/unet.yaml`
- **nnU-Net**: Uses its own automated preprocessing pipeline (2D trainer)
- **MedSAM**: Uses its own image preparation (1024×1024 resize for ViT encoder)
- **Rationale**: Forcing identical preprocessing would invalidate nnU-Net's core method

### 3. Common Evaluation Space
All predictions are resampled to the original image resolution before metric computation. This ensures metrics are comparable regardless of each model's internal resolution.

### 4. Identical Metrics
A single metrics implementation (`src/evaluation/metrics.py`) computes Dice, IoU, and HD95 for all models. No model uses a different metric library.

### 5. Seed Control
- Training seeds: [11, 22, 33] for three repeated runs (U-Net and MedSAM fine-tuning)
- Framework settings: `torch.manual_seed()`, `numpy.random.seed()`, `random.seed()`, `torch.cuda.manual_seed_all()`
- CUDA determinism: `torch.backends.cudnn.deterministic = True`, `torch.backends.cudnn.benchmark = False`
- nnU-Net: Uses nnU-Net's internal seed mechanism

## MedSAM Prompt Protocol

### Mode
To be declared in `configs/medsam.yaml`: either `fine_tuned` or `zero_shot`.

### Prompt Type
Bounding box prompts.

### Prompt Generation Rules
1. **Training prompts**: Derived from ground-truth mask bounding boxes with optional padding (e.g., 10-20 pixels)
2. **Inference prompts**: Same generation method applied to either:
   - Ground-truth boxes (if evaluating segmentation quality only)
   - Automatically detected boxes (if evaluating full pipeline)
3. **Determinism**: Same input mask always produces the same bounding box prompt
4. **Documentation**: Exact padding, jitter, and derivation method recorded in `configs/medsam.yaml`

## Execution Policy

### Run Order
1. Phase 1: Create configs, split, protocol (this document)
2. Phase 2: Build evaluation scaffold
3. Phase 3: U-Net train → infer → evaluate (3 seeds)
4. Phase 4: nnU-Net convert → train → infer → evaluate
5. Phase 5: MedSAM configure → train/load → infer → evaluate
6. Phase 6: Aggregate results → create report

### Artifact Storage
```
artifacts/
├── unet/
│   └── predictions/
│       ├── seed_11/
│       ├── seed_22/
│       └── seed_33/
├── nnunet/
│   ├── predictions/
│   └── run_manifest.json
├── medsam/
│   └── predictions/
│       ├── seed_11/ (or mode/)
│       ├── seed_22/
│       └── seed_33/
└── summary/
    ├── case_metrics.json
    ├── model_case_metrics.json
    ├── model_summary.json
    ├── run_summary.json
    ├── run_inventory.json
    └── figures/
```

### Reporting Format
- Per-case metrics in JSON with run metadata, threshold policy, lesion counts, and foreground volumes
- Summary tables serialized as JSON at both run level and model level
- Distribution plots for Dice plus boundary metrics (HD95 and ASSD)
- Qualitative overlay gallery for best, median, and worst cases
- Final narrative report in markdown following the medical experiment reporting template

## Version Control
- All configurations committed to git before any training
- Exact library versions recorded in environment file
- Git commit hash recorded per run in `run_registry.json`
