# train_fixed.py
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms, datasets
from tqdm import tqdm

# -------------- CONFIG -----------------
DATASET_DIR = "dataset/training"
VAL_DIR = "dataset/validation"
MODEL_OUT = "bus_classifier_resnet18.pth"
BATCH = 16
EPOCHS = 12
LR = 1e-4
IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 0  # use 0 no Windows; pode aumentar em Linux
# ---------------------------------------

def make_transforms(img_size):
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomResizedCrop(img_size, scale=(0.85,1.0)),
        transforms.ToTensor()
    ])
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])
    return train_tf, val_tf

def train_loop():
    train_tf, val_tf = make_transforms(IMG_SIZE)

    # datasets e dataloaders criados dentro da main (importante no Windows)
    train_ds = datasets.ImageFolder(DATASET_DIR, transform=train_tf)
    val_ds = datasets.ImageFolder(VAL_DIR, transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=NUM_WORKERS)

    # Model: ResNet18 pretrained
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 2)  # binary classes
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        # ------------------ TRAIN ------------------
        model.train()
        total = 0
        correct = 0
        total_loss = 0.0

        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} - train"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            _, pred = torch.max(out, 1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total if total > 0 else 0.0
        train_loss = total_loss / total if total > 0 else 0.0

        # ------------------ VALIDATION ------------------
        model.eval()
        total = 0
        correct = 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                out = model(imgs)
                _, pred = torch.max(out, 1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total if total > 0 else 0.0

        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")

        # save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_OUT)
            print(f"*** Melhor modelo salvo: {MODEL_OUT} (acc={val_acc:.4f})")

def main():
    # opcional: torch.multiprocessing.set_start_method('spawn')  # geralmente padrão no Windows
    train_loop()

if __name__ == "__main__":
    main()
