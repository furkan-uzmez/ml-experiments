import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from tqdm import tqdm
import argparse
import os
import sys

# Add project root to path to import local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from functions.logger import get_logger
try:
    from PadChest_evaluation.dataset import PadChestBinaryDataset
except ImportError:
    PadChestBinaryDataset = None

def train(model, train_loader, val_loader, criterion, optimizer, device, save_path, num_epochs=5, patience=3, log_path='training.log'):
    logger = get_logger(log_path)
    logger.info("Training started.")

    def log_print(message):
        print(message)
        logger.info(message)
            
    model.to(device)

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    best_epoch = 0
    best_val_loss = float('inf')  
    epochs_without_improvement = 0 
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0

        # --- EĞİTİM ---
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels).item()
            total_samples += labels.size(0)
            
            pbar.set_postfix({'loss': running_loss / (pbar.n + 1)})

        avg_train_loss = running_loss / len(train_loader)
        train_accuracy = running_corrects / total_samples

        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)

        # --- DOĞRULAMA ---
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels).item()
                val_total += labels.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = val_corrects / val_total

        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)

        log_print(f"Epoch [{epoch+1}/{num_epochs}]")
        log_print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}")
        log_print(f"  Val   Loss: {avg_val_loss:.4f}, Val   Acc: {val_accuracy:.4f}")

        # --- EARLY STOPPING KONTROLÜ ---
        if avg_val_loss < best_val_loss:
            best_epoch = epoch + 1
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            log_print(f"  Best model saved with val_loss: {best_val_loss:.4f}")
            epochs_without_improvement = 0  # Reset
        else:
            epochs_without_improvement += 1
            log_print(f"  No improvement. Early stopping counter: {epochs_without_improvement}/{patience}")

            if epochs_without_improvement >= patience:
                log_print(f"  Early stopping triggered. Training stopped.")
                break

    # --- EN İYİ MODEL SONUÇLARI LOGA EKLENİR ---
    log_print("\n========== BEST MODEL SUMMARY ==========")
    log_print(f"Best Epoch      : {best_epoch}")
    log_print(f"Val Loss        : {best_val_loss:.4f}")
    log_print("========================================")

    return train_losses, train_accuracies, val_losses, val_accuracies

def main():
    parser = argparse.ArgumentParser(description="Train a CNN on PadChest dataset")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to image directory")
    parser.add_argument("--csv_file", type=str, required=True, help="Path to CSV file")
    parser.add_argument("--model", type=str, default="resnet18", help="Model architecture (resnet18, resnet50, densenet121)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    parser.add_argument("--save_path", type=str, default="best_model.pth", help="Path to save the best model")
    parser.add_argument("--log_path", type=str, default="training.log", help="Path to log file")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of classes")
    
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Dataset
    if PadChestBinaryDataset is None:
        print("Error: Could not import PadChestBinaryDataset from PadChest_evaluation.dataset")
        return

    print("Loading datasets...")
    train_dataset = PadChestBinaryDataset(args.csv_file, args.data_dir, transform=transform, split='train')
    val_dataset = PadChestBinaryDataset(args.csv_file, args.data_dir, transform=transform, split='validation')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Model
    print(f"Initializing {args.model}...")
    if args.model == "resnet18":
        model = models.resnet18(weights='DEFAULT')
        model.fc = nn.Linear(model.fc.in_features, args.num_classes)
    elif args.model == "resnet50":
        model = models.resnet50(weights='DEFAULT')
        model.fc = nn.Linear(model.fc.in_features, args.num_classes)
    elif args.model == "densenet121":
        model = models.densenet121(weights='DEFAULT')
        model.classifier = nn.Linear(model.classifier.in_features, args.num_classes)
    else:
        print(f"Error: Architecture {args.model} not supported in CLI (yet).")
        return

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        save_path=args.save_path,
        num_epochs=args.epochs,
        patience=args.patience,
        log_path=args.log_path
    )

if __name__ == "__main__":
    main()

