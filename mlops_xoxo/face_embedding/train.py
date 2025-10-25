#!/usr/bin/env python3
"""
Train face embedding model with MobileNetV2 + ArcFaceHead.
Logs experiments via MLflow and emissions via CodeCarbon.
"""

import os
import random
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from tqdm import tqdm
from PIL import Image

import mlflow
import mlflow.pytorch
from dotenv import load_dotenv

from train_util import (
    start_emissions_tracker,
    log_metrics_mlflow,
    log_params_mlflow,
    prepare_output_dirs,
)

# ============================================================
# CONFIG + ENV SETUP
# ============================================================

with open("pipelines/face_embedding/params.yaml", encoding="utf-8") as f:
    params = yaml.safe_load(f)

# --- Load environment variables for remote MLflow ---
load_dotenv()
mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
if mlflow_uri:
    mlflow.set_tracking_uri(mlflow_uri)

mlflow_username = os.getenv("MLFLOW_TRACKING_USERNAME")
mlflow_password = os.getenv("MLFLOW_TRACKING_PASSWORD")
if mlflow_username and mlflow_password:
    os.environ["MLFLOW_TRACKING_USERNAME"] = mlflow_username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = mlflow_password

run_name = params["mlflow"].get("run_name", "default_run")

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

DATA_DIR = Path(params["dataset"]["processed_dir"]) / "train"
VAL_DIR = Path(params["dataset"]["processed_dir"]) / "val"

model_dir, report_dir = prepare_output_dirs("face_embedding")
EMISSIONS_OUTPUT_PATH = report_dir / "emissions.csv"


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

def train_model(run_id=None):
    NUM_CLASSES = len(train_ds.classes)
    model = MobileFace(emb_size=512).to(DEVICE)
    arcface = ArcFaceHead(emb_size=512, num_classes=NUM_CLASSES, margin=MARGIN).to(DEVICE)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(arcface.parameters()), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = torch.nn.CrossEntropyLoss()

    log_params_mlflow({
        "batch_size": BATCH,
        "epochs": EPOCHS,
        "lr": LR,
        "margin": MARGIN,
        "weight_decay": WEIGHT_DECAY,
        "num_classes": NUM_CLASSES,
        "device": str(DEVICE),
    })

    best_val_acc, best_val_loss, best_epoch = 0.0, float("inf"), 0
    print("\nStarting CodeCarbon Emissions Tracker...")

    with start_emissions_tracker("face_embedding", report_dir):
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
            log_metrics_mlflow({"train_loss": avg_loss, "val_loss": val_loss, "val_accuracy": val_acc}, step=epoch)

            print(f"Epoch {epoch}: train_loss={avg_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}")
            if val_acc > best_val_acc:
                best_val_acc, best_val_loss, best_epoch = val_acc, val_loss, epoch

        # Final metrics
        mlflow.log_metrics({
            "final_train_loss": avg_loss,
            "final_val_loss": val_loss,
            "final_val_accuracy": val_acc,
            "best_val_accuracy": best_val_acc,
        })
        mlflow.log_param("best_epoch", best_epoch)

        # Log emissions data
        try:
            df = pd.read_csv(EMISSIONS_OUTPUT_PATH).tail(1)
            mlflow.log_metric("carbon_emissions_kg", df["emissions"].iloc[0])
            mlflow.log_metric("energy_kwh", df["energy_consumed"].iloc[0])
        except Exception as e:
            print(f"⚠️ Emission log failed: {e}")

    # Save and register model
    model_path = model_dir / "mobilenetv2_arcface_model.pth"
    torch.save(model.state_dict(), model_path)
    mlflow.log_artifact(str(model_path))
    mlflow.pytorch.log_model(model, artifact_path="pytorch_model", registered_model_name="mobilenetv2_arcface_model")

    print(f"✅ Training complete. Best val acc = {best_val_acc:.4f}")


# ============================================================
# MAIN ENTRYPOINT (Your preferred structure)
# ============================================================

if __name__ == "__main__":
    # Get experiment id or create one
    experiment_id = params["mlflow"].get("experiment_id")
    if not experiment_id:
        experiment_name = params["mlflow"].get("experiment_name", "face_embedding")
        mlflow.set_experiment(experiment_name)
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id
        params["mlflow"]["experiment_id"] = experiment_id
        with open("pipelines/fake_classification/params.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(params, f)
    else:
        mlflow.set_experiment(params["mlflow"].get("experiment_name", "face_embedding"))

    # Start top-level run with experiment_id
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as parent:
        params["mlflow"]["run_id"] = parent.info.run_id
        with open("pipelines/face_embedding/params.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(params, f)

        # Nested run for training
        with mlflow.start_run(nested=True, run_name="train_model"):
            train_model(run_id=mlflow.active_run().info.run_id)