#!/usr/bin/env python
# coding: utf-8

# # Gradient Accumulation Experiment
# 
# Bu notebook, **standart egitim** ile **gradient accumulation** egitimini ayni efektif batch size altinda karsilastirir.
# 
# Karsilastirilan metrikler:
# - Loss ve Accuracy
# - Toplam/epoch/step sureleri
# - Throughput (samples/sec, batches/sec)
# - Donanim kullanimi (GPU VRAM, CPU, RAM)

# In[1]:


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


# In[2]:


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


# In[3]:


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


# In[4]:


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


# ## Standart Egitim

# In[5]:


standard_result = run_experiment(
    batch_size=STANDARD['batch_size'],
    accumulation_steps=STANDARD['accumulation_steps'],
    run_name=STANDARD['run_name'],
)
standard_result


# ## Gradient Accumulation Egitimi

# In[6]:


accum_result = run_experiment(
    batch_size=ACCUM['batch_size'],
    accumulation_steps=ACCUM['accumulation_steps'],
    run_name=ACCUM['run_name'],
)
accum_result


# In[7]:


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


# In[8]:


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


# In[9]:


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


# ## Sonuc Notlari
# 
# Bu hucrede iki kosunun trade-off analizini kisa maddeler halinde yazin:
# - Hangi kosu daha hizli?
# - Hangi kosu daha az VRAM kullandi?
# - Accuracy/Loss farki anlamli mi?
# - Donanim kisitli ortamlarda hangi ayar tercih edilmeli?
