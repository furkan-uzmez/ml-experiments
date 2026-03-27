# Medical Image Segmentation Benchmark

## What This Is

A deterministic benchmark comparing U-Net, nnU-Net, and MedSAM on a single medical image segmentation dataset under a controlled and reproducible protocol. The benchmark standardizes split policy, preprocessing, compute budget, metrics, and reporting so that observed performance differences are attributable to model behavior rather than inconsistent experimental setup.

## Core Value

Fair, reproducible comparison of three segmentation models on the same dataset with identical evaluation conditions — results must be trustworthy and attributable to model differences, not experimental artifacts.

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] Evaluate exactly three models: U-Net, nnU-Net, and MedSAM
- [ ] Use one dataset only for the primary benchmark
- [ ] Use a fixed patient-level train/validation/test split for all models
- [ ] Preserve the same test set across all runs and reruns
- [ ] Report Dice, IoU, HD95, inference time per case, and peak GPU memory
- [ ] Store predictions for the complete test set for all three models
- [ ] Use the same random seeds for all models where the framework permits seed control
- [ ] Produce a single comparison table and narrative summary
- [ ] Run at least three repeated training runs for U-Net and MedSAM to estimate variance
- [ ] Run nnU-Net with prescribed automatic configuration without manual architecture changes
- [ ] Treat MedSAM as either zero-shot or fine-tuned with consistently declared protocol
- [ ] Use the same class definitions and label mapping for all three models
- [ ] Use the same resampled evaluation space for all three models before metric computation
- [ ] Remove or avoid storing direct patient identifiers in benchmark artifacts

### Out of Scope

- Forcing identical preprocessing for all models — nnU-Net needs its own automatic pipeline
- Single-run comparison only — stochastic variance needs multiple runs for reliability
- Slice-level random splitting — risks patient leakage and inflated scores
- Zero-shot-only evaluation for MedSAM — would underrepresent dataset-adapted performance

## Context

- **Domain**: Medical image segmentation (modality TBD based on dataset)
- **Models**: U-Net (supervised baseline), nnU-Net (automated strong baseline), MedSAM (foundation model)
- **Evaluation**: Shared evaluation pipeline with Dice, IoU, HD95, latency, and peak GPU memory
- **Seeds**: Fixed seed list [11, 22, 33] for reproducible runs
- **Fairness**: Enforced at data split and evaluation protocol layers, not by forcing identical internal pipelines
- **MedSAM prompts**: Must be deterministic and documented

## Constraints

- **Compute**: Sufficient GPU memory needed for nnU-Net planning/training and MedSAM inference or fine-tuning
- **Dependencies**: nnU-Net installation and preprocessing tooling, MedSAM checkpoints and prompt encoder
- **Data**: Patient-level identifiers must be available or derivable from the dataset
- **nnU-Net autonomy**: nnU-Net uses its own preprocessing and patch strategy internally; cannot force identical internals
- **MedSAM prompts**: Prompt generation rules must be deterministic and documented

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Patient-level splitting (not slice-level) | Prevents data leakage between train/test | — Pending |
| nnU-Net uses its own preprocessing | Forcing identical preprocessing would invalidate its core method | — Pending |
| Three repeated runs with seeds [11,22,33] | Needed to estimate stochastic variance for U-Net and MedSAM | — Pending |
| Shared evaluation space for all models | Ensures metric comparability across different internal pipelines | — Pending |
| MedSAM protocol must be explicitly declared | Prevents ambiguity in zero-shot vs fine-tuned comparison | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-03-27 after initialization*
