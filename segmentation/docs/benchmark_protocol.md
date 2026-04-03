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
All three models (U-Net, nnU-Net, MedSAM3) use the exact same train/val/test patient assignments from `primary_split.json`.

### 2. Independent Preprocessing
- **U-Net**: Standardized resize + normalization defined in `configs/unet.yaml`
- **nnU-Net**: Uses its own automated preprocessing pipeline (2D trainer)
- **MedSAM3**: Uses its own SAM3 text-guided preprocessing (1008×1008 resize)
- **Rationale**: Forcing identical preprocessing would invalidate nnU-Net's core method

### 3. Common Evaluation Space
All predictions are resampled to the original image resolution before metric computation. This ensures metrics are comparable regardless of each model's internal resolution.

### 4. Identical Metrics
A single metrics implementation (`src/evaluation/metrics.py`) computes Dice, IoU, and HD95 for all models. No model uses a different metric library.

### 5. Seed Control
- Training seeds: [11, 22, 33] for three repeated runs (U-Net and any future MedSAM3 fine-tuning)
- Framework settings: `torch.manual_seed()`, `numpy.random.seed()`, `random.seed()`, `torch.cuda.manual_seed_all()`
- CUDA determinism: `torch.backends.cudnn.deterministic = True`, `torch.backends.cudnn.benchmark = False`
- nnU-Net: Uses nnU-Net's internal seed mechanism

## MedSAM3 Prompt Protocol

### Runtime Source
- Official upstream repository: `https://github.com/Joey-S-Liu/MedSAM3`
- Default checkout location for this benchmark: `segmentation/external/MedSAM3`
- Override mechanism: set `MEDSAM3_REPO=/absolute/path/to/MedSAM3` when using a different checkout

### Mode
To be declared in `configs/medsam.yaml`: either `fine_tuned` or `zero_shot`.

### Prompt Type
Text prompts.

### Prompt Generation Rules
1. **Inference prompts**: Declared explicitly in `configs/medsam.yaml` as medical concepts (for example `skin lesion`, `lesion`)
2. **Prompt set**: Multiple synonymous prompts may be evaluated per image and merged into one binary prediction mask
3. **Determinism**: The same prompt list, score threshold, mask threshold, and NMS rule must be reused for every test image
4. **Documentation**: Exact prompts and thresholds are recorded in `configs/medsam.yaml`

## Execution Policy

### Run Order
1. Phase 1: Create configs, split, protocol (this document)
2. Phase 2: Build evaluation scaffold
3. Phase 3: U-Net train → infer → evaluate (3 seeds)
4. Phase 4: nnU-Net convert → train → infer → evaluate
5. Phase 5: MedSAM3 configure → load → infer → evaluate
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
├── medsam3/
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
