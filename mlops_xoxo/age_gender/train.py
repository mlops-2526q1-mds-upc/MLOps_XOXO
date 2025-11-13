#!/usr/bin/env python3
"""
Entraînement GenderAge avec modèle PyTorch natif
"""
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2, numpy as np, pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse
import yaml
with open("pipelines/age_gender/params.yaml", encoding="utf-8") as f:
    params = yaml.safe_load(f)

train_dir = params["dataset"]["processed_dir"] + "/train"
val_dir = params["dataset"]["processed_dir"] + "/val"
model_dir = "models/age_gender"

import torch
import torch.nn as nn

class GenderAgeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(512, 128), nn.ReLU(inplace=True), nn.Dropout(0.3)
        )
        self.gender_head = nn.Linear(128, 2)
        self.age_head = nn.Linear(128, 1)

    def forward(self, x):
        x = self.features(x).view(x.size(0), -1)
        x = self.classifier(x)
        gender = self.gender_head(x)
        age = self.age_head(x)
        return torch.cat([gender, age], dim=1)


class GenderAgeLoss(nn.Module):
    def __init__(self, gender_weight=1.0, age_weight=1.0):
        super().__init__()
        self.gender_loss = nn.CrossEntropyLoss()
        self.age_loss = nn.MSELoss()
        self.gender_weight = gender_weight
        self.age_weight = age_weight

    def forward(self, predictions, gender_target, age_target):
        gender_pred = predictions[:, :2]
        age_pred = predictions[:, 2]
        loss_gender = self.gender_loss(gender_pred, gender_target)
        loss_age = self.age_loss(age_pred, age_target)
        total_loss = self.gender_weight * loss_gender + self.age_weight * loss_age
        return total_loss, loss_gender, loss_age
    
class GenderAgeDataset(Dataset):
    def __init__(self, csv_path, images_dir, input_size=(112, 112)):
        self.df = pd.read_csv(csv_path)
        self.images_dir = Path(images_dir)
        self.input_size = input_size
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.imread(str(self.images_dir / row['image_name']))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.input_size).astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)
        gender = torch.tensor(row['gender'], dtype=torch.long)
        age = torch.tensor(row['age'] / 100.0, dtype=torch.float32)
        return img, gender, age


def train_epoch(model, loader, criterion, optimizer, device):
    model.train(); total_loss, correct, total = 0, 0, 0
    for imgs, genders, ages in tqdm(loader, desc="Training"):
        imgs, genders, ages = imgs.to(device), genders.to(device), ages.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss, _, _ = criterion(outputs, genders, ages)
        loss.backward(); optimizer.step()
        total_loss += loss.item()
        correct += (torch.argmax(outputs[:, :2], 1) == genders).sum().item()
        total += genders.size(0)
    return total_loss/len(loader), correct/total


def validate(model, loader, criterion, device):
    model.eval(); total_loss, correct, total, age_err = 0, 0, 0, []
    with torch.no_grad():
        for imgs, genders, ages in tqdm(loader, desc="Validation"):
            imgs, genders, ages = imgs.to(device), genders.to(device), ages.to(device)
            out = model(imgs)
            loss, _, _ = criterion(out, genders, ages)
            total_loss += loss.item()
            correct += (torch.argmax(out[:, :2], 1) == genders).sum().item()
            total += genders.size(0)
            age_err.extend((out[:, 2]*100 - ages*100).abs().cpu().numpy())
    return total_loss/len(loader), correct/total, np.mean(age_err)


def train_model(train_loader, val_loader, epochs, device, save_dir):
    model = GenderAgeModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    criterion = GenderAgeLoss(1.0, 2.0)
    best_val = float('inf')
    save_dir = Path(save_dir); save_dir.mkdir(exist_ok=True)

    for ep in range(1, epochs+1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_mae = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        print(f"\nEpoch {ep}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.2%}, MAE={val_mae:.2f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), save_dir / 'best_genderage_native.pth')
            print("✓ Nouveau meilleur modèle sauvegardé")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-data', default=train_dir)
    parser.add_argument('--val-data', default=val_dir)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output-dir', default=model_dir)
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() or args.device == "mps" else "cpu"
    train_ds = GenderAgeDataset(Path(args.train_data)/'labels.csv', Path(args.train_data)/'images')
    val_ds = GenderAgeDataset(Path(args.val_data)/'labels.csv', Path(args.val_data)/'images')
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    train_model(train_loader, val_loader, args.epochs, device, args.output_dir)