import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from src.metrics import calculate_metrics
from src.utils import EpochTimer, get_gpu_memory_usage, reset_memory_stats
import numpy as np

def train_epoch(model, dataloader, criterion, optimizer, precision, scaler=None, device='cuda'):
    model.train()
    running_loss = 0.0
    all_targets = []
    all_preds = []
    all_probs = []

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        if precision == 'fp32':
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
        elif precision == 'fp16':
            # Pure FP16
            with autocast(device_type=device, dtype=torch.float16):
                outputs = model(inputs.half()) 
                loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
        elif precision == 'amp':
            # Mixed Precision
            with autocast(device_type=device, dtype=torch.float16):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            raise ValueError("precision must be 'fp32', 'fp16', or 'amp'")

        running_loss += loss.item() * inputs.size(0)
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)
        
        all_targets.extend(targets.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.detach().cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    metrics = calculate_metrics(np.array(all_targets), np.array(all_probs), np.array(all_preds))
    return epoch_loss, metrics

@torch.no_grad()
def evaluate(model, dataloader, criterion, precision, device='cuda'):
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_preds = []
    all_probs = []

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

        if precision == 'fp32':
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        elif precision == 'fp16':
            with autocast(device_type=device, dtype=torch.float16):
                outputs = model(inputs.half())
                loss = criterion(outputs, targets)
        elif precision == 'amp':
            with autocast(device_type=device, dtype=torch.float16):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

        running_loss += loss.item() * inputs.size(0)
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)

        all_targets.extend(targets.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.detach().cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    metrics = calculate_metrics(np.array(all_targets), np.array(all_probs), np.array(all_preds))
    return epoch_loss, metrics

def train_model(model, train_loader, val_loader, precision='fp32', epochs=5, lr=1e-4, device='cuda'):
    model = model.to(device)
    if precision == 'fp16':
        model = model.half()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    scaler = GradScaler() if precision == 'amp' else None

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'val_auc': [], 'val_f1': [], 'epoch_times': []}
    
    reset_memory_stats()
    timer = EpochTimer()

    for epoch in range(epochs):
        timer.start()
        
        train_loss, train_metrics = train_epoch(model, train_loader, criterion, optimizer, precision, scaler, device)
        val_loss, val_metrics = evaluate(model, val_loader, criterion, precision, device)
        
        epoch_time = timer.stop()
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_auc'].append(val_metrics['auc_roc'])
        history['val_f1'].append(val_metrics['f1_score'])
        history['epoch_times'].append(epoch_time)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val AUC: {val_metrics['auc_roc']:.4f} - Time: {epoch_time:.2f}s")
        
    peak_memory = get_gpu_memory_usage()
    print(f"Peak GPU Memory for {precision}: {peak_memory:.2f} MB")
    
    return model, history, peak_memory
