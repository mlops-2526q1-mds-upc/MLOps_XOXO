#!/usr/bin/env python3
"""
Train face embedding model with MobileNetV2 + ArcFaceHead.
CI/CD version without MLflow tracking.
"""

import os
import random
import numpy as np
from pathlib import Path
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from tqdm import tqdm
from PIL import Image

# ============================================================
# CONFIG + ENV SETUP
# ============================================================

with open("pipelines/face_embedding/params.yaml", encoding="utf-8") as f:
    params = yaml.safe_load(f)

# --- Device setup (CUDA / MPS / CPU) ---
device_param = params["training"]["device"].lower()
if device_param == "cuda" and torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif device_param == "mps" and getattr(torch.backends, "mps", None) is not None:
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")

# ============================================================
# PARAMS + PATHS
# ============================================================

BATCH = params["training"]["batch_size"]
EPOCHS = params["training"]["epochs"]
LR = float(params["training"]["lr"])
MARGIN = float(params["training"]["margin"])
WEIGHT_DECAY = float(params["training"]["weight_decay"])

DATA_DIR = "data/processed/face_embedding/train"
VAL_DIR = "data/processed/face_embedding/val"

# Create model directory
model_dir = Path("models/face_embedding")
model_dir.mkdir(parents=True, exist_ok=True)

# ============================================================
# DATASET
# ============================================================

class FaceDataset(Dataset):
    """Dataset and triplet sampling"""
    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.samples = []
        self.classes = sorted([p.name for p in self.root.iterdir() if p.is_dir()])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        for c in self.classes:
            for img in (self.root / c).glob("*.jpg"):
                self.samples.append((str(img), self.class_to_idx[c]))
        self.class_indices = {}
        for idx, (_, label) in enumerate(self.samples):
            self.class_indices.setdefault(label, []).append(idx)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

train_ds = FaceDataset(DATA_DIR, transform=transform)
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0)
val_ds = FaceDataset(VAL_DIR, transform=transform)
val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=0)
print(f"Loaded {len(train_ds.classes)} classes from dataset.")

# ============================================================
# MODEL
# ============================================================

class MobileFace(torch.nn.Module):
    def __init__(self, emb_size=512):
        super().__init__()
        backbone = models.mobilenet_v2(weights="IMAGENET1K_V1")
        self.backbone = backbone.features
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Linear(backbone.last_channel, emb_size)
        self.bn = torch.nn.BatchNorm1d(emb_size)

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x).flatten(1)
        x = self.fc(x)
        x = self.bn(x)
        return F.normalize(x)


class ArcFaceHead(torch.nn.Module):
    def __init__(self, emb_size, num_classes, scale=64.0, margin=0.5):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.FloatTensor(num_classes, emb_size))
        torch.nn.init.xavier_uniform_(self.weight)
        self.scale = scale
        self.margin = margin

    def forward(self, embeddings, labels):
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        theta = torch.acos(torch.clamp(cosine, -1 + 1e-7, 1 - 1e-7))
        target_logit = torch.cos(theta + self.margin)
        one_hot = F.one_hot(labels, num_classes=cosine.size(1)).float()
        logits = cosine * (1 - one_hot) + target_logit * one_hot
        return logits * self.scale


# ============================================================
# TRAIN FUNCTION
# ============================================================

def train_model():
    NUM_CLASSES = len(train_ds.classes)
    model = MobileFace(emb_size=512).to(DEVICE)
    arcface = ArcFaceHead(emb_size=512, num_classes=NUM_CLASSES, margin=MARGIN).to(DEVICE)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(arcface.parameters()),
                                 lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_acc, best_val_loss, best_epoch = 0.0, float("inf"), 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            emb = model(images)
            logits = arcface(emb, labels)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                emb = model(images)
                logits = arcface(emb, labels)
                loss = criterion(logits, labels)
                val_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_loss /= len(val_loader)
        val_acc = correct / total

        print(f"Epoch {epoch}: train_loss={avg_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc, best_val_loss, best_epoch = val_acc, val_loss, epoch
            # Save best model
            model_path = model_dir / "mobilenetv2_arcface_model.pth"
            torch.save(model.state_dict(), model_path)
            print(f"  ✓ Saved best model to {model_path}")

    print(f"\nTraining complete. Best val acc = {best_val_acc:.4f} at epoch {best_epoch}")


# ============================================================
# MAIN ENTRYPOINT
# ============================================================

if __name__ == "__main__":
    train_model()
