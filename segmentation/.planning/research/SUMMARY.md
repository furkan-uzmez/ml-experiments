# Research Summary — Medical Image Segmentation Benchmark

## Key Findings

### Stack
- **Framework**: PyTorch ≥ 2.2 (all three models are PyTorch-native)
- **nnU-Net**: v2 (≥ 2.5) — self-configuring, automated preprocessing and training
- **MedSAM**: ViT-B encoder + mask decoder, from segment-anything foundation
- **U-Net**: MONAI implementation recommended (validated, medical-focused)
- **I/O**: SimpleITK + nibabel for medical image formats
- **Metrics**: medpy + surface-distance (DeepMind) for Dice, HD95

### Table Stakes
- Patient-level data splitting (prevents leakage)
- Fixed random seeds across all frameworks
- Common evaluation space with resampling
- Per-case + aggregate metrics (Dice, IoU, HD95)
- Multiple runs for variance estimation
- Inference time and peak GPU memory measurement

### Watch Out For
1. **🔴 Data leakage via slice splitting** — Can inflate Dice by 5-15%
2. **🔴 Unfair evaluation space** — Models at different resolutions = invalid HD95
3. **🟡 MedSAM prompt bias** — Prompt quality drastically affects results
4. **🟡 nnU-Net fold mismatch** — Must use benchmark split, not default 5-fold
5. **🟡 Metric implementation inconsistencies** — Edge cases (empty predictions)
6. **🟠 Class imbalance masking** — Average metrics hide per-class failures
7. **🟠 Seed non-determinism** — CUDA operations may not be fully deterministic

### Architecture
- 5 layers: Config → Data → Models → Evaluation → Reporting
- Adapter pattern for nnU-Net (translates between benchmark and nnU-Net formats)
- Shared evaluation pipeline: all models produce predictions → resample → compute metrics
- Build order: Config → Data/Split → Evaluation → U-Net → nnU-Net → MedSAM → Reporting

### Critical Design Decisions
- nnU-Net keeps its internal preprocessing; fairness enforced at evaluation layer
- MedSAM prompts must be deterministic with documented generation rules
- All predictions stored in standardized format under `artifacts/<model>/predictions/`
- Configuration files are immutable after setup

## Files
- `.planning/research/STACK.md` — Technology stack recommendations
- `.planning/research/FEATURES.md` — Feature categories and scoping
- `.planning/research/ARCHITECTURE.md` — System design and data flow
- `.planning/research/PITFALLS.md` — Domain-specific risks and prevention
