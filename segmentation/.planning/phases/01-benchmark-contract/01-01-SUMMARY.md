# Phase 1 Summary: Benchmark Contract & Data Split

## Completed
All Phase 1 deliverables created and committed.

## Artifacts Created

| File | Purpose |
|------|---------|
| `configs/dataset.yaml` | ISIC 2018 Task 1 contract (path, modality, labels, normalization) |
| `configs/seeds.yaml` | Training seeds [11, 22, 33] + CUDA determinism settings |
| `configs/metrics.yaml` | Dice, IoU, HD95, latency, peak GPU memory with edge-case handling |
| `splits/primary_split.json` | Patient-level split: 1815 train / 389 val / 390 test |
| `docs/benchmark_protocol.md` | Fairness rules, MedSAM prompt protocol, execution policy |

## Key Decisions
- **Split from training set only**: Validation/test sets in the original ISIC 2018 lack GT masks, so we split the 2594 training images
- **70/15/15 ratio**: Provides 390 test cases for stable metric estimation
- **Patient = image**: Each ISIC image is from a unique patient (no multi-image patients)
- **Split seed = 42**: Fixed, deterministic, reproducible

## Verification
- ✓ Zero overlap between train/val/test verified programmatically
- ✓ Total count matches: 1815 + 389 + 390 = 2594
- ✓ All config files valid YAML
- ✓ Protocol document covers all fairness rules
