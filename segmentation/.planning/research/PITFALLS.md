# Pitfalls Research — Medical Image Segmentation Benchmark

## Critical Pitfalls

### 1. Data Leakage via Slice-Level Splitting
- **Risk Level**: 🔴 Critical
- **Warning Signs**: Unusually high test performance; model generalizes poorly to external datasets
- **Prevention**: Split ONLY at patient level. Verify `splits/primary_split.json` has zero patient overlap across train/val/test
- **When to Address**: Phase 1 (Benchmark Contract & Split)
- **Details**: In volumetric medical imaging, adjacent slices from the same patient are highly correlated. If slices are randomly assigned to train/test, the model memorizes patient-specific features rather than learning generalizable patterns. This can inflate Dice scores by 5-15%.

### 2. Unfair Evaluation Space
- **Risk Level**: 🔴 Critical
- **Warning Signs**: Models with different internal resolutions score differently on same cases; inconsistent HD95 values
- **Prevention**: Resample ALL predictions and ALL ground truth to a single common evaluation space before any metric computation
- **When to Address**: Phase 2 (Evaluation Scaffold)
- **Details**: nnU-Net may output predictions at a different spacing than U-Net or MedSAM. Without resampling to a common space, metric comparisons are invalid because voxel sizes affect distance-based metrics like HD95.

### 3. MedSAM Prompt Bias
- **Risk Level**: 🟡 High
- **Warning Signs**: MedSAM results vary wildly between runs despite fixed seeds; MedSAM underperforms or overperforms unexpectedly
- **Prevention**: Use deterministic prompt generation from ground-truth bounding boxes for training, and a documented automatic method for inference. Record exact prompt parameters.
- **When to Address**: Phase 5 (MedSAM Pipeline)
- **Details**: MedSAM's performance is highly sensitive to prompt quality. Using poorly designed prompts (too loose, too tight, off-center) can make MedSAM results misleading. For fair comparison, prompt generation must be deterministic and documented.

### 4. nnU-Net Fold Mismatch with Primary Split
- **Risk Level**: 🟡 High
- **Warning Signs**: nnU-Net uses different patients for validation than other models
- **Prevention**: Configure nnU-Net to use the benchmark's primary split (same train/val/test patients), not its default 5-fold cross-validation split
- **When to Address**: Phase 4 (nnU-Net Pipeline)
- **Details**: nnU-Net's default behavior is 5-fold cross-validation on the training set. For a fair 3-model comparison, nnU-Net must use the same fixed validation and test sets as U-Net and MedSAM.

### 5. Metric Implementation Inconsistencies
- **Risk Level**: 🟡 High
- **Warning Signs**: Repeating metrics with different libraries yields different results; Dice of empty predictions not handled
- **Prevention**: Use a single metric implementation for all models. Test on synthetic masks with known expected values. Handle edge cases (empty predictions, empty ground truth) explicitly.
- **When to Address**: Phase 2 (Evaluation Scaffold)
- **Details**: Different Dice implementations handle edge cases differently (e.g., both prediction and ground truth empty: Dice = 1 or undefined?). HD95 implementations may differ in how they handle the 95th percentile.

### 6. Class Imbalance Masking Failures
- **Risk Level**: 🟠 Medium
- **Warning Signs**: Average metrics look good but small-structure segmentation is poor; large background class dominates
- **Prevention**: Report per-class metrics separately. Use appropriate loss functions (Dice loss, focal loss) during training. Include worst-case analysis.
- **When to Address**: Phase 2 (Metrics definition) + Phase 6 (Reporting)

### 7. GPU Memory Inconsistencies
- **Risk Level**: 🟠 Medium
- **Warning Signs**: Models trained with different batch sizes leading to different convergence behavior
- **Prevention**: Document GPU model and VRAM. Use consistent batch size where possible. If batch sizes must differ, document the reason and its potential impact.
- **When to Address**: Phase 3, 4, 5 (All model training phases)

### 8. Seed Non-Determinism
- **Risk Level**: 🟠 Medium
- **Warning Signs**: Same seed produces different results across runs
- **Prevention**: Set `torch.manual_seed()`, `torch.cuda.manual_seed_all()`, `numpy.random.seed()`, `random.seed()`, `torch.backends.cudnn.deterministic = True`, `torch.backends.cudnn.benchmark = False`
- **When to Address**: Phase 1 (Seeds configuration) + All training phases
- **Details**: Even with fixed seeds, CUDA operations may not be fully deterministic. Document any residual non-determinism.

### 9. Small Test Set Instability
- **Risk Level**: 🟠 Medium
- **Warning Signs**: HD95 variance is very high; ranking changes with removal of a single case
- **Prevention**: Use a test set with ≥20 cases if possible. Report confidence intervals. Consider bootstrap sampling for stability analysis.
- **When to Address**: Phase 1 (Split design) + Phase 6 (Reporting)

## Prevention Checklist

- [ ] Patient-level split with zero overlap verified by automated test
- [ ] Common evaluation space defined and resampling validated
- [ ] Single metric library used for all models
- [ ] Edge cases in Dice/HD95 explicitly handled and tested
- [ ] MedSAM prompts documented with deterministic generation
- [ ] nnU-Net uses benchmark's primary split (not its own 5-fold)
- [ ] All seeds set across PyTorch, NumPy, Python random
- [ ] Per-class metrics reported alongside aggregate metrics
- [ ] GPU configuration documented
- [ ] Test set size justification provided
