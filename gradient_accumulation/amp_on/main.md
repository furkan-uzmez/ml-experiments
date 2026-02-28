# Gradient Accumulation Experiment

Bu notebook, **standart egitim** ile **gradient accumulation** egitimini ayni efektif batch size altinda karsilastirir.

Karsilastirilan metrikler:
- Loss ve Accuracy
- Toplam/epoch/step sureleri
- Throughput (samples/sec, batches/sec)
- Donanim kullanimi (GPU VRAM, CPU, RAM)


```python
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

from functions.dataset import (
    COVIDCXNetDataset,
    DataLoaderConfig,
    build_transforms,
    create_dataloader,
    describe_class_distribution,
)
from functions.logging import ExperimentLogger, ExperimentLoggerConfig
from functions.train import TrainConfig, fit, set_seed

sns.set_theme(style='whitegrid')
```


```python
CONFIG = {
    'csv_file': '/home/furkan/Documents/covidx/covidx_merged.csv',
    'root_dir': '/home/furkan/Documents',
    'num_epochs': 5,
    'num_classes': 2,
    'lr': 1e-4,
    'seed': 42,
    'patience': 3,
    'image_size': 224,
    'output_dir': 'runs',
    'num_workers': None,
}

STANDARD = {
    'batch_size': 32,
    'accumulation_steps': 1,
    'run_name': 'standard_bs32_acc1',
}

ACCUM = {
    'batch_size': 8,
    'accumulation_steps': 4,
    'run_name': 'accum_bs8_acc4',
}

assert (
    STANDARD['batch_size'] * STANDARD['accumulation_steps']
    == ACCUM['batch_size'] * ACCUM['accumulation_steps']
), 'Effective batch size must be identical for a fair comparison.'

set_seed(CONFIG['seed'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
print(f'Effective batch size: {STANDARD["batch_size"] * STANDARD["accumulation_steps"]}')
```

    Using device: cuda
    Effective batch size: 32



```python
train_transform = build_transforms(image_size=CONFIG['image_size'], augment=True)
eval_transform = build_transforms(image_size=CONFIG['image_size'], augment=False)

train_dataset = COVIDCXNetDataset(
    csv_file=CONFIG['csv_file'],
    root_dir=CONFIG['root_dir'],
    transform=train_transform,
    split='train',
)

val_dataset = None
for split_name in ('val', 'validation', 'valid', 'test'):
    try:
        val_dataset = COVIDCXNetDataset(
            csv_file=CONFIG['csv_file'],
            root_dir=CONFIG['root_dir'],
            transform=eval_transform,
            split=split_name,
        )
        print(f'Validation split selected: {split_name}')
        break
    except ValueError:
        continue

if val_dataset is None:
    raise ValueError('No validation split found. Update split names in this cell.')

print('Train class distribution:', describe_class_distribution(train_dataset))
print('Val class distribution:', describe_class_distribution(val_dataset))
```

    Validation split selected: val
    Train class distribution: {'AP': 33303, 'PA': 20388}
    Val class distribution: {'AP': 2911, 'PA': 1275}



```python
def build_model(num_classes: int) -> nn.Module:
    model = models.resnet18(weights='DEFAULT')
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def run_experiment(batch_size: int, accumulation_steps: int, run_name: str):
    train_loader = create_dataloader(
        train_dataset,
        DataLoaderConfig(
            batch_size=batch_size,
            shuffle=True,
            num_workers=CONFIG['num_workers'],
            drop_last=False,
        ),
        device=device,
    )

    val_loader = create_dataloader(
        val_dataset,
        DataLoaderConfig(
            batch_size=batch_size,
            shuffle=False,
            num_workers=CONFIG['num_workers'],
            drop_last=False,
        ),
        device=device,
    )

    model = build_model(CONFIG['num_classes'])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'])

    logger = ExperimentLogger(
        ExperimentLoggerConfig(
            run_name=run_name,
            output_dir=CONFIG['output_dir'],
            overwrite=True,
        )
    )

    train_cfg = TrainConfig(
        num_epochs=CONFIG['num_epochs'],
        accumulation_steps=accumulation_steps,
        patience=CONFIG['patience'],
        run_name=run_name,
        output_dir=CONFIG['output_dir'],
        log_every_n_steps=10,
        system_log_interval=50,
    )

    result = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        config=train_cfg,
        logger=logger,
        show_progress=True,
    )
    return result
```

## Standart Egitim


```python
standard_result = run_experiment(
    batch_size=STANDARD['batch_size'],
    accumulation_steps=STANDARD['accumulation_steps'],
    run_name=STANDARD['run_name'],
)
standard_result
```

    /home/furkan/Projects/ML_Algorithms/gradient_accumulation_codex/functions/train.py:318: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
      scaler = GradScaler(enabled=amp_enabled)
    [2026-02-26 02:17:19] Run 'standard_bs32_acc1' started | epochs=5, accumulation_steps=1, amp=True
    Epoch 1 [train]:   0%|          | 0/1678 [00:00<?, ?it/s]/home/furkan/Projects/ML_Algorithms/gradient_accumulation_codex/functions/train.py:170: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast(enabled=amp_enabled):
    [2026-02-26 02:19:31] Epoch 1/5 | train_loss=0.0996, train_acc=0.9679, val_loss=0.1169, val_acc=0.9646, epoch_time=122.43s, train_sps=438.54, peak_vram=1702.54MB
    Epoch 2 [train]:   0%|          | 0/1678 [00:00<?, ?it/s]/home/furkan/Projects/ML_Algorithms/gradient_accumulation_codex/functions/train.py:170: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast(enabled=amp_enabled):
    [2026-02-26 02:21:40] Epoch 2/5 | train_loss=0.0644, train_acc=0.9806, val_loss=0.1103, val_acc=0.9666, epoch_time=118.97s, train_sps=451.29, peak_vram=573.32MB
    Epoch 3 [train]:   0%|          | 0/1678 [00:00<?, ?it/s]/home/furkan/Projects/ML_Algorithms/gradient_accumulation_codex/functions/train.py:170: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast(enabled=amp_enabled):
    [2026-02-26 02:23:48] Epoch 3/5 | train_loss=0.0568, train_acc=0.9834, val_loss=0.1413, val_acc=0.9541, epoch_time=118.68s, train_sps=452.40, peak_vram=573.32MB
    Epoch 4 [train]:   0%|          | 0/1678 [00:00<?, ?it/s]/home/furkan/Projects/ML_Algorithms/gradient_accumulation_codex/functions/train.py:170: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast(enabled=amp_enabled):
    [2026-02-26 02:25:55] Epoch 4/5 | train_loss=0.0503, train_acc=0.9852, val_loss=0.1022, val_acc=0.9675, epoch_time=117.95s, train_sps=455.22, peak_vram=573.32MB
    Epoch 5 [train]:   0%|          | 0/1678 [00:00<?, ?it/s]/home/furkan/Projects/ML_Algorithms/gradient_accumulation_codex/functions/train.py:170: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast(enabled=amp_enabled):
    [2026-02-26 02:28:03] Epoch 5/5 | train_loss=0.0457, train_acc=0.9860, val_loss=0.1024, val_acc=0.9680, epoch_time=118.09s, train_sps=454.65, peak_vram=573.32MB
    [2026-02-26 02:28:03] Run 'standard_bs32_acc1' finished in 643.40s





    TrainingResult(run_name='standard_bs32_acc1', run_dir='runs/standard_bs32_acc1', history={'train_loss': [0.09956893286732482, 0.06435090115935349, 0.05678534453972014, 0.0503137866273264, 0.045691464586817145], 'train_accuracy': [0.96794621072433, 0.9806485258236949, 0.9833864148553761, 0.9851744240189231, 0.9859939282188821], 'val_loss': [0.11689214291776553, 0.1103203687481462, 0.14134019607659368, 0.10216872903319053, 0.1024083290991097], 'val_accuracy': [0.9646440516005733, 0.9665551839464883, 0.9541328236980411, 0.9675107501194458, 0.9679885332059245], 'epoch_time_sec': [122.43182563500886, 118.97347364800225, 118.68122538700118, 117.94568226698902, 118.09422478399938], 'avg_step_time_sec': [0.03242119225739182, 0.03101087442307817, 0.030899076715667932, 0.03089873048850507, 0.030922259280656304], 'train_samples_per_sec': [438.53793506324456, 451.2854702288635, 452.3967445139021, 455.2180204313184, 454.6454333241418], 'train_batches_per_sec': [13.70558669117961, 14.10398426261446, 14.138714818020295, 14.22688790083538, 14.20899288740962], 'peak_vram_mb': [1702.5380859375, 573.32275390625, 573.32275390625, 573.32275390625, 573.32275390625]}, best_epoch=4, best_val_loss=0.10216872903319053, best_checkpoint_path='runs/standard_bs32_acc1/best_model.pth', total_train_time_sec=643.4040210370003, accumulation_steps=1)



## Gradient Accumulation Egitimi


```python
accum_result = run_experiment(
    batch_size=ACCUM['batch_size'],
    accumulation_steps=ACCUM['accumulation_steps'],
    run_name=ACCUM['run_name'],
)
accum_result
```

    /home/furkan/Projects/ML_Algorithms/gradient_accumulation_codex/functions/train.py:318: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
      scaler = GradScaler(enabled=amp_enabled)
    [2026-02-26 02:28:03] Run 'accum_bs8_acc4' started | epochs=5, accumulation_steps=4, amp=True
    Epoch 1 [train]:   0%|          | 0/6712 [00:00<?, ?it/s]/home/furkan/Projects/ML_Algorithms/gradient_accumulation_codex/functions/train.py:170: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast(enabled=amp_enabled):
    [2026-02-26 02:30:18] Epoch 1/5 | train_loss=0.1272, train_acc=0.9596, val_loss=0.1170, val_acc=0.9642, epoch_time=125.81s, train_sps=426.77, peak_vram=1354.29MB
    Epoch 2 [train]:   0%|          | 0/6712 [00:00<?, ?it/s]/home/furkan/Projects/ML_Algorithms/gradient_accumulation_codex/functions/train.py:170: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast(enabled=amp_enabled):
    [2026-02-26 02:32:31] Epoch 2/5 | train_loss=0.0750, train_acc=0.9787, val_loss=0.1038, val_acc=0.9642, epoch_time=123.78s, train_sps=433.77, peak_vram=323.38MB
    Epoch 3 [train]:   0%|          | 0/6712 [00:00<?, ?it/s]/home/furkan/Projects/ML_Algorithms/gradient_accumulation_codex/functions/train.py:170: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast(enabled=amp_enabled):
    [2026-02-26 02:34:43] Epoch 3/5 | train_loss=0.0643, train_acc=0.9826, val_loss=0.1244, val_acc=0.9606, epoch_time=123.60s, train_sps=434.40, peak_vram=323.38MB
    Epoch 4 [train]:   0%|          | 0/6712 [00:00<?, ?it/s]/home/furkan/Projects/ML_Algorithms/gradient_accumulation_codex/functions/train.py:170: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast(enabled=amp_enabled):
    [2026-02-26 02:36:56] Epoch 4/5 | train_loss=0.0575, train_acc=0.9839, val_loss=0.1029, val_acc=0.9661, epoch_time=123.60s, train_sps=434.40, peak_vram=323.38MB
    Epoch 5 [train]:   0%|          | 0/6712 [00:00<?, ?it/s]/home/furkan/Projects/ML_Algorithms/gradient_accumulation_codex/functions/train.py:170: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast(enabled=amp_enabled):
    [2026-02-26 02:39:09] Epoch 5/5 | train_loss=0.0533, train_acc=0.9853, val_loss=0.0939, val_acc=0.9709, epoch_time=123.58s, train_sps=434.47, peak_vram=323.38MB
    [2026-02-26 02:39:09] Run 'accum_bs8_acc4' finished in 665.90s





    TrainingResult(run_name='accum_bs8_acc4', run_dir='runs/accum_bs8_acc4', history={'train_loss': [0.12721202636652745, 0.07496352233129336, 0.06426538274038218, 0.05752520092427414, 0.05327986487265042], 'train_accuracy': [0.959639418152018, 0.9786556406101581, 0.9825669106554171, 0.9838892924326237, 0.9852675494961912], 'val_loss': [0.11701469051010997, 0.10383905176891689, 0.12442877926856859, 0.1028564729693542, 0.09389842581248638], 'val_accuracy': [0.9641662685140946, 0.9641662685140946, 0.960582895365504, 0.9660774008600096, 0.9708552317247969], 'epoch_time_sec': [125.80679553499795, 123.77694186700683, 123.59855910700571, 123.5990796660044, 123.57770808199712], 'avg_step_time_sec': [0.0159452934653792, 0.015488818833411829, 0.015211419518926178, 0.015083274792970478, 0.015288335295082368], 'train_samples_per_sec': [426.7734486970841, 433.77222922253765, 434.3982679726622, 434.39643842888233, 434.4715631428816], 'train_batches_per_sec': [53.351649022272426, 54.226578058551205, 54.304840189836455, 54.304611475566816, 54.314002939319835], 'peak_vram_mb': [1354.2861328125, 323.37841796875, 323.37841796875, 323.37841796875, 323.37841796875]}, best_epoch=5, best_val_loss=0.09389842581248638, best_checkpoint_path='runs/accum_bs8_acc4/best_model.pth', total_train_time_sec=665.8991306399985, accumulation_steps=4)




```python
def load_run_tables(result):
    run_dir = Path(result.run_dir)
    step_df = pd.read_csv(run_dir / 'step_metrics.csv')
    epoch_df = pd.read_csv(run_dir / 'epoch_metrics.csv')
    system_df = pd.read_csv(run_dir / 'system_metrics.csv')
    summary = pd.read_json(run_dir / 'run_summary.json', typ='series')
    return step_df, epoch_df, system_df, summary


std_step, std_epoch, std_sys, std_summary = load_run_tables(standard_result)
acc_step, acc_epoch, acc_sys, acc_summary = load_run_tables(accum_result)


def summarize_run(name: str, epoch_df: pd.DataFrame, system_df: pd.DataFrame, summary: pd.Series):
    train_epoch = epoch_df[epoch_df['phase'] == 'train'].copy()
    val_epoch = epoch_df[epoch_df['phase'] == 'val'].copy()

    row = {
        'run': name,
        'best_epoch': summary.get('best_epoch', np.nan),
        'best_val_loss': summary.get('best_val_loss', np.nan),
        'total_train_time_sec': summary.get('total_train_time_sec', np.nan),
        'final_train_loss': train_epoch['loss'].iloc[-1] if len(train_epoch) else np.nan,
        'final_train_acc': train_epoch['accuracy'].iloc[-1] if len(train_epoch) else np.nan,
        'final_val_loss': val_epoch['loss'].iloc[-1] if len(val_epoch) else np.nan,
        'final_val_acc': val_epoch['accuracy'].iloc[-1] if len(val_epoch) else np.nan,
        'avg_epoch_time_sec': train_epoch['epoch_time_sec'].mean() if len(train_epoch) else np.nan,
        'avg_step_time_sec': train_epoch['avg_step_time_sec'].mean() if len(train_epoch) else np.nan,
        'avg_samples_per_sec': train_epoch['samples_per_sec'].mean() if len(train_epoch) else np.nan,
        'avg_batches_per_sec': train_epoch['batches_per_sec'].mean() if len(train_epoch) else np.nan,
        'peak_vram_mb': train_epoch['peak_vram_mb'].max() if len(train_epoch) else np.nan,
        'max_cpu_percent': system_df['cpu_percent'].max() if len(system_df) else np.nan,
        'max_ram_percent': system_df['ram_percent'].max() if len(system_df) else np.nan,
    }
    return row


comparison_df = pd.DataFrame([
    summarize_run('Standard', std_epoch, std_sys, std_summary),
    summarize_run('GradientAccum', acc_epoch, acc_sys, acc_summary),
])
comparison_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>run</th>
      <th>best_epoch</th>
      <th>best_val_loss</th>
      <th>total_train_time_sec</th>
      <th>final_train_loss</th>
      <th>final_train_acc</th>
      <th>final_val_loss</th>
      <th>final_val_acc</th>
      <th>avg_epoch_time_sec</th>
      <th>avg_step_time_sec</th>
      <th>avg_samples_per_sec</th>
      <th>avg_batches_per_sec</th>
      <th>peak_vram_mb</th>
      <th>max_cpu_percent</th>
      <th>max_ram_percent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Standard</td>
      <td>4</td>
      <td>0.102169</td>
      <td>643.404021</td>
      <td>0.045691</td>
      <td>0.985994</td>
      <td>0.102408</td>
      <td>0.967989</td>
      <td>119.225286</td>
      <td>0.031230</td>
      <td>450.416721</td>
      <td>14.076833</td>
      <td>1702.538086</td>
      <td>69.4</td>
      <td>78.3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>GradientAccum</td>
      <td>5</td>
      <td>0.093898</td>
      <td>665.899131</td>
      <td>0.053280</td>
      <td>0.985268</td>
      <td>0.093898</td>
      <td>0.970855</td>
      <td>124.071817</td>
      <td>0.015403</td>
      <td>432.762389</td>
      <td>54.100336</td>
      <td>1354.286133</td>
      <td>69.3</td>
      <td>70.8</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for label, epoch_df in [('Standard', std_epoch), ('GradientAccum', acc_epoch)]:
    train_df = epoch_df[epoch_df['phase'] == 'train']
    val_df = epoch_df[epoch_df['phase'] == 'val']

    axes[0, 0].plot(train_df['epoch'], train_df['loss'], marker='o', label=f'{label} Train')
    if len(val_df):
        axes[0, 0].plot(val_df['epoch'], val_df['loss'], marker='s', linestyle='--', label=f'{label} Val')

    axes[0, 1].plot(train_df['epoch'], train_df['accuracy'], marker='o', label=f'{label} Train')
    if len(val_df):
        axes[0, 1].plot(val_df['epoch'], val_df['accuracy'], marker='s', linestyle='--', label=f'{label} Val')

axes[0, 0].set_title('Loss Curves')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()

axes[0, 1].set_title('Accuracy Curves')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Accuracy')
axes[0, 1].legend()

sns.barplot(data=comparison_df, x='run', y='total_train_time_sec', ax=axes[1, 0])
axes[1, 0].set_title('Total Training Time (sec)')
axes[1, 0].set_xlabel('Run')
axes[1, 0].set_ylabel('Seconds')

bar_data = comparison_df.melt(
    id_vars=['run'],
    value_vars=['avg_samples_per_sec', 'peak_vram_mb'],
    var_name='metric',
    value_name='value',
)
sns.barplot(data=bar_data, x='run', y='value', hue='metric', ax=axes[1, 1])
axes[1, 1].set_title('Throughput and Peak VRAM')
axes[1, 1].set_xlabel('Run')
axes[1, 1].set_ylabel('Value')

plt.tight_layout()
plt.show()
```


    
![png](main_files/main_10_0.png)
    



```python
hardware_df = comparison_df[[
    'run',
    'peak_vram_mb',
    'max_cpu_percent',
    'max_ram_percent',
    'avg_epoch_time_sec',
    'avg_step_time_sec',
    'avg_samples_per_sec',
    'avg_batches_per_sec',
]]
hardware_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>run</th>
      <th>peak_vram_mb</th>
      <th>max_cpu_percent</th>
      <th>max_ram_percent</th>
      <th>avg_epoch_time_sec</th>
      <th>avg_step_time_sec</th>
      <th>avg_samples_per_sec</th>
      <th>avg_batches_per_sec</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Standard</td>
      <td>1702.538086</td>
      <td>69.4</td>
      <td>78.3</td>
      <td>119.225286</td>
      <td>0.031230</td>
      <td>450.416721</td>
      <td>14.076833</td>
    </tr>
    <tr>
      <th>1</th>
      <td>GradientAccum</td>
      <td>1354.286133</td>
      <td>69.3</td>
      <td>70.8</td>
      <td>124.071817</td>
      <td>0.015403</td>
      <td>432.762389</td>
      <td>54.100336</td>
    </tr>
  </tbody>
</table>
</div>



## Sonuc Notlari

Bu hucrede iki kosunun trade-off analizini kisa maddeler halinde yazin:
- Hangi kosu daha hizli?
- Hangi kosu daha az VRAM kullandi?
- Accuracy/Loss farki anlamli mi?
- Donanim kisitli ortamlarda hangi ayar tercih edilmeli?
