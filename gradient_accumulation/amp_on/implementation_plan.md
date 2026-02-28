# Gradient Accumulation Experiment - Implementation Plan

## 1. Objective
Design and implement a controlled ML experiment that compares:
- Standard training (`batch_size=32`, `accumulation_steps=1`)
- Gradient accumulation training (`batch_size=8`, `accumulation_steps=4`)

Both runs must be compared with identical effective batch size and measured on:
- Loss and accuracy
- Total training time, average epoch time, average step time
- Throughput (samples/sec and batches/sec)
- Hardware usage (max GPU VRAM, CPU usage, RAM usage)

This plan is written before code changes and will drive refactoring in:
- `functions/dataset.py`
- `functions/train.py`
- `functions/evaluation.py`
- `functions/logger.py` (and alignment with requested `functions/logging.py`)
- `main.ipynb` (new)


## 2. Skills-Driven Workflow (From `.agent/skills`)
I will use the following skills as the implementation backbone:

1. `pytorch`  
Reason: Core training-loop and data-loader best practices, gradient accumulation correctness, device/memory patterns.

2. `pytorch/reference/debugging.md`  
Reason: Accurate timing, memory tracking, leak prevention, and profiler-compatible instrumentation.

3. `ipynb-notebooks` + `ipynb-notebooks/references/presentation-patterns.md`  
Reason: Notebook flow design, reproducibility, and clean analysis/reporting structure.

4. `13-mlops/tensorboard` (optional integration layer)  
Reason: Structured scalar logging and optional profiler traces for experiment comparison.

### Workflow order
1. Audit current module behavior and identify correctness/performance gaps.
2. Refactor data pipeline for stable high-throughput loading.
3. Refactor training loop to support standard + accumulation in a single validated path.
4. Refactor evaluation for correct, leak-safe metric computation.
5. Refactor logging to persist timing/throughput/hardware + metrics to disk.
6. Build `main.ipynb` with ordered analysis flow and comparative plots.
7. Run smoke validation and consistency checks.


## 3. Current State Audit (What must be fixed)

## `functions/dataset.py`
- Dataset class is tied to specific columns and prints extensively.
- No dedicated `DataLoader` factory for `num_workers`, `pin_memory`, `prefetch_factor`, `persistent_workers`.
- No explicit handling for performance-oriented settings by device/runtime.
- No caching strategy and no reusable preprocessing configuration object.

## `functions/train.py`
- Loop currently supports only standard optimization steps.
- Gradient accumulation path is missing.
- `optimizer.zero_grad()` usage is not optimized (`set_to_none=True` absent).
- No strict separation of `train_one_epoch` vs `fit`.
- No structured per-step/per-epoch timing or throughput collection.
- No VRAM peak tracking lifecycle (`reset_peak_memory_stats` + readout).

## `functions/evaluation.py`
- Evaluation logic mixes plotting and metric computation.
- Should use leak-safe inference wrapper and return machine-readable metrics.
- Metric aggregation should be batch-size weighted and consistent with training logging.

## `functions/logger.py`
- Minimal logger exists, but no experiment metrics persistence for:
  - step/epoch durations
  - throughput
  - VRAM peak
  - CPU/RAM stats
- No structured file outputs (CSV/JSON) for notebook comparison pipeline.

## Project root
- `main.ipynb` does not exist yet.
- No experiment orchestration notebook for side-by-side comparison.


## 4. Refactor Plan by Module

## Step A - `functions/dataset.py`
Target: fast, reproducible, configurable data input path.

Planned changes:
1. Add a clear dataset config interface:
   - paths, split, transforms, cache flag, image mode, class mapping.
2. Keep dataset class focused on:
   - CSV read/filter
   - path resolve
   - sample decode
   - transform apply
3. Add optional in-memory cache mode for repeated reads (small/medium dataset scenarios).
4. Add `create_dataloader(...)` helper with:
   - `num_workers` auto/default logic
   - `pin_memory` conditional on CUDA
   - `persistent_workers=True` when workers > 0
   - `prefetch_factor` when workers > 0
5. Ensure robust preprocessing defaults:
   - resize
   - tensor conversion
   - normalization
6. Add complete docstrings and typed signatures.

Validation checks:
- DataLoader starts without deadlocks.
- Batch tensor shapes are consistent.
- Effective input pipeline speed improves vs current baseline.


## Step B - `functions/logger.py` and requested `functions/logging.py` alignment
Target: durable experiment telemetry for comparison plots/tables.

Planned changes:
1. Introduce structured logging utilities (CSV + JSON), including:
   - `step_metrics.csv`
   - `epoch_metrics.csv`
   - `system_metrics.csv`
   - `run_summary.json`
2. Track per-step fields:
   - epoch, global_step
   - loss, accuracy (if available at step scope)
   - step_time_sec
   - samples_per_sec, batches_per_sec
   - lr
3. Track per-epoch fields:
   - train/val loss and accuracy
   - epoch_time_sec
   - avg_step_time_sec
   - peak_vram_mb
4. Track system fields:
   - cpu_percent
   - ram_used_mb / ram_percent
   - gpu memory allocated/reserved/max (if CUDA)
5. Keep backward compatibility for imports from `functions.logger`.
   - If needed, add a `logging.py` module and keep `logger.py` as compatibility shim.
6. Add complete docstrings and stable output schema.


## Step C - `functions/train.py`
Target: one correct, optimized training path that supports both scenarios.

Planned changes:
1. Split responsibilities:
   - `train_one_epoch(...)`
   - `validate_one_epoch(...)`
   - `fit(...)` (orchestrator)
2. Add gradient accumulation support:
   - configurable `accumulation_steps`
   - scale loss by effective accumulation factor before backward
   - optimizer step only at accumulation boundary (and epoch tail remainder)
3. Gradient best practices:
   - `optimizer.zero_grad(set_to_none=True)` at correct boundaries
   - optional grad clipping
   - optional AMP (`autocast` + `GradScaler`) for CUDA
4. Timing and throughput:
   - synchronize CUDA around timers where needed
   - capture step time and epoch time
   - compute samples/sec and batches/sec
5. Hardware tracking:
   - `torch.cuda.reset_peak_memory_stats()` per epoch on CUDA
   - log `torch.cuda.max_memory_allocated()` after epoch
   - sample CPU/RAM via `psutil`
6. Reproducibility controls:
   - seed setup helper
   - deterministic flags option
7. Return a clean run artifact object for notebook consumption:
   - history dict/dataframe path references
   - best checkpoint metadata

Correctness invariants to enforce:
- Standard mode and accumulation mode differ only in micro-batch mechanics.
- Effective batch size parity is preserved.
- Optimizer step count is logged and verified.


## Step D - `functions/evaluation.py`
Target: leak-safe and correct evaluation metrics.

Planned changes:
1. Use `torch.inference_mode()` (or `no_grad`) + `model.eval()`.
2. Move all tensors needed for metrics to CPU safely (`detach().cpu()`).
3. Compute loss/accuracy with proper sample weighting.
4. Keep optional advanced metrics (precision/recall/F1/AUC) but return as structured dict.
5. Separate plotting helpers from metric computation helpers.
6. Add docstrings and clear contracts for expected model outputs.

Validation checks:
- No gradient graph retained after evaluation.
- Repeated eval runs do not increase GPU memory footprint unexpectedly.


## 5. Notebook Plan - `main.ipynb` (new)
Notebook sections will follow the requested analysis flow:

1. Setup
   - imports
   - seed config
   - device detection
   - experiment directories
2. Data Loading
   - dataset + dataloader creation
   - class distribution and sample sanity checks
3. Standard Training Run
   - config: `batch_size=32`, `accumulation_steps=1`
   - run training + validation
   - persist logs
4. Gradient Accumulation Run
   - config: `batch_size=8`, `accumulation_steps=4`
   - run training + validation
   - persist logs
5. Comparative Analysis
   - table: final/best metrics
   - plots (matplotlib/seaborn):
     - train/val loss curves
     - accuracy curves
     - epoch time and avg step time bars
     - throughput comparison
     - peak VRAM + CPU/RAM comparison
6. Conclusions
   - concise interpretation and trade-off summary
   - notes on reproducibility and caveats

Notebook quality rules (from skill guidance):
- Keep reusable logic in `functions/`, not in notebook cells.
- Keep outputs concise and deterministic.
- Ensure "Restart & Run All" executes end-to-end.


## 6. Experiment Design Details

Control variables (kept constant):
- Model architecture and initialization seed
- Optimizer, LR scheduler, loss function
- Number of epochs and dataset splits
- Effective batch size (32 in both setups)

Compared variables:
- micro-batch size
- accumulation steps

Primary metrics to report:
- `train_loss`, `val_loss`, `train_accuracy`, `val_accuracy`
- `total_train_time_sec`, `avg_epoch_time_sec`, `avg_step_time_sec`
- `samples_per_sec`, `batches_per_sec`
- `peak_vram_mb`, `cpu_percent`, `ram_used_mb`


## 7. Verification and Acceptance Criteria

Functional:
1. Both training modes run successfully with same effective batch size.
2. Evaluation returns correct metrics without gradient tracking leaks.
3. Logs are written to disk in a consistent schema.
4. Notebook executes from top to bottom and generates comparison artifacts.

Quality:
1. All public functions include complete docstrings.
2. Module boundaries are clear and reusable.
3. Plot outputs are legible and directly comparable across runs.


## 8. Delivery Sequence
1. Refactor `dataset.py`.
2. Refactor logging module (`logger.py` and `logging.py` compatibility plan).
3. Refactor `train.py`.
4. Refactor `evaluation.py`.
5. Create `main.ipynb`.
6. Run smoke tests and finalize artifacts.

