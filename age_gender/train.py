"""
train.py - Train gender and age models with MLflow tracking
"""

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
import yaml

# MLflow imports
from config import MLFLOW_TRACKING_URI
import mlflow
import mlflow.pytorch

# Load parameters
with open('params.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Configuration from params.yaml
DATA_DIR = Path("data/raw/utkface_aligned_cropped/UTKFace")
TRAIN_FILE = Path("data/processed/splits/train.csv")
VAL_FILE = Path("data/processed/splits/val.csv")
OUTPUT_DIR = Path("models")

IMAGE_SIZE = config['training']['image_size']
BATCH_SIZE = config['training']['batch_size']
NUM_EPOCHS = config['training']['num_epochs']
LEARNING_RATE = config['training']['learning_rate']
SEED = config['training']['seed']

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configure MLflow
if MLFLOW_TRACKING_URI:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    print(f"✅ MLflow tracking: {MLFLOW_TRACKING_URI}")

# Set experiment
mlflow.set_experiment(config['experiment']['name'])


class UTKFaceDataset(Dataset):
    """Simple dataset loader"""
    def __init__(self, df, data_dir, transform, task):
        self.df = df.reset_index(drop=True)
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.task = task
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(self.data_dir / row['filename']).convert('RGB')
        img = self.transform(img)
        
        if self.task == 'gender':
            label = torch.tensor(row['gender'], dtype=torch.long)
        else:
            label = torch.tensor(row['age'], dtype=torch.float32)
        
        return img, label


class SimpleModel(nn.Module):
    """Lightweight MobileNetV2-based model"""
    def __init__(self, num_outputs):
        super().__init__()
        self.backbone = models.mobilenet_v2(weights='IMAGENET1K_V1')
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=config['model']['dropout']),
            nn.Linear(self.backbone.last_channel, num_outputs)
        )
    
    def forward(self, x):
        out = self.backbone(x)
        return out.squeeze() if out.shape[1] == 1 else out


def get_transforms():
    """Basic transforms"""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def train_model(task, train_df, val_df):
    """Train a single model with MLflow tracking"""
    print(f"\n{'='*60}")
    print(f"TRAINING: {task.upper()}")
    print(f"{'='*60}")
    
    # Start MLflow run
    with mlflow.start_run(run_name=f"{task}_training") as run:
        
        # Log parameters
        mlflow.log_params({
            "task": task,
            "batch_size": BATCH_SIZE,
            "num_epochs": NUM_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "image_size": IMAGE_SIZE,
            "backbone": config['model']['backbone'],
            "dropout": config['model']['dropout'],
            "device": str(DEVICE),
            "seed": SEED
        })
        
        # Log tags
        mlflow.set_tags(config['experiment']['tags'])
        mlflow.set_tag("model_type", task)
        
        # Create datasets
        train_dataset = UTKFaceDataset(train_df, DATA_DIR, get_transforms(), task)
        val_dataset = UTKFaceDataset(val_df, DATA_DIR, get_transforms(), task)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        
        # Create model
        num_outputs = 2 if task == 'gender' else 1
        model = SimpleModel(num_outputs).to(DEVICE)
        
        criterion = nn.CrossEntropyLoss() if task == 'gender' else nn.L1Loss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        # Training loop
        best_metric = 0 if task == 'gender' else float('inf')
        
        for epoch in range(NUM_EPOCHS):
            # Train
            model.train()
            train_loss = 0
            for images, labels in train_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validate
            model.eval()
            val_loss = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(DEVICE), labels.to(DEVICE)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    
                    if task == 'gender':
                        preds = torch.argmax(outputs, dim=1)
                        correct += (preds == labels).sum().item()
                    else:
                        correct += torch.abs(outputs - labels).sum().item()
                    
                    total += labels.size(0)
            
            val_loss /= len(val_loader)
            metric = correct / total if task == 'gender' else correct / total
            
            # Log metrics to MLflow
            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss,
                f"val_{'accuracy' if task == 'gender' else 'mae'}": metric
            }, step=epoch)
            
            print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Loss: {train_loss:.4f} - Val: {val_loss:.4f} - Metric: {metric:.4f}")
            
            # Save best
            is_best = (metric > best_metric) if task == 'gender' else (metric < best_metric)
            if is_best:
                best_metric = metric
                print(f"  -> Best model saved!")
        
        # Log best metric
        metric_name = "best_accuracy" if task == 'gender' else "best_mae"
        mlflow.log_metric(metric_name, best_metric)
        
        # Save final model
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        model_path = OUTPUT_DIR / f"{task}_model.pt"
        torch.save(model.state_dict(), model_path)
        print(f"\nSaved to: {model_path}")
        
        # Log model to MLflow
        mlflow.pytorch.log_model(model, f"{task}_model")
        
        # Log model file as artifact
        mlflow.log_artifact(str(model_path))
        
        print(f"✅ MLflow Run ID: {run.info.run_id}")


def main():
    print("=" * 60)
    print(f"MODEL TRAINING (Device: {DEVICE})")
    print("=" * 60)
    
    torch.manual_seed(SEED)
    
    # Load data
    train_df = pd.read_csv(TRAIN_FILE)
    val_df = pd.read_csv(VAL_FILE)
    print(f"\nTrain: {len(train_df)}, Val: {len(val_df)}")
    
    # Train both models (separate MLflow runs)
    train_model('gender', train_df, val_df)
    train_model('age', train_df, val_df)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    exit(main())