# State — Medical Image Segmentation Benchmark

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-27)

**Core value:** Fair, reproducible comparison of U-Net, nnU-Net, MedSAM on ISIC 2018 with identical evaluation
**Current focus:** Project initialization complete — ready for Phase 1 planning

## Current Position

- **Milestone**: v1.0 — Single-Dataset Benchmark
- **Phase**: Pre-execution (initialization complete)
- **Next action**: `/gsd-plan-phase 1` → Benchmark Contract & Data Split

## Recent Progress

| Date | Action | Outcome |
|------|--------|---------|
| 2026-03-27 | Project initialized | PROJECT.md, config.json created |
| 2026-03-27 | Domain research completed | 4 research docs + summary |
| 2026-03-27 | Requirements defined | 35 requirements across 7 categories |
| 2026-03-27 | Dataset selected | ISIC 2018 Task 1 (Lesion Segmentation) |
| 2026-03-27 | Roadmap created | 6 phases with requirement mappings |

## Key Decisions

- Dataset: ISIC 2018 Task 1 (2D dermoscopy, binary segmentation)
- nnU-Net: 2D trainer mode
- MedSAM: Bounding-box prompts from ground truth
- Evaluation: Common space resampling before all metrics
- Seeds: [11, 22, 33] for all stochastic runs

## Session Continuity

Last session ended: 2026-03-27
Context: Project fully initialized, ready for phase planning
