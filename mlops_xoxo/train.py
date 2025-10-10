# train.py
import os
from dotenv import load_dotenv
import yaml
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
from PIL import Image
import random
from tqdm import tqdm
import mlflow
import mlflow.pytorch
import pandas as pd
from codecarbon import EmissionsTracker

with open("params.yaml") as f:
    params = yaml.safe_load(f)

load_dotenv()
mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
if mlflow_uri:
    mlflow.set_tracking_uri(mlflow_uri)
mlflow_username = os.getenv("MLFLOW_TRACKING_USERNAME")
mlflow_password = os.getenv("MLFLOW_TRACKING_PASSWORD")
if mlflow_username and mlflow_password:
    os.environ["MLFLOW_TRACKING_USERNAME"] = mlflow_username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = mlflow_password

run_name = params['mlflow'].get('run_name', 'default_run')


# Device setup
device_param = params['training']['device'].lower()
if device_param == 'cuda' and torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif device_param == 'mps' and getattr(torch.backends, 'mps', None) is not None:
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')
print("Using device:", DEVICE)

BATCH = params['training']['batch_size']
EPOCHS = params['training']['epochs']
LR = float(params['training']['lr'])
MARGIN = float(params['training']['margin'])
WEIGHT_DECAY = float(params['training']['weight_decay'])

DATA_DIR = Path(params['dataset']['processed_dir']) / 'train'
VAL_DIR = Path(params['dataset']['processed_dir']) / 'val'
# Dataset and triplet sampling
class FaceDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.samples = []
        self.classes = sorted([p.name for p in self.root.iterdir() if p.is_dir()])
        self.class_to_idx = {c:i for i,c in enumerate(self.classes)}
        for c in self.classes:
            for img in (self.root / c).glob('*.jpg'):
                self.samples.append((str(img), self.class_to_idx[c]))
        # index per class
        self.class_indices = {}
        for idx, (_, label) in enumerate(self.samples):
            self.class_indices.setdefault(label, []).append(idx)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

    def sample_triplet(self):
        anchor_label = random.choice(list(self.class_indices.keys()))
        anchor_idx = random.choice(self.class_indices[anchor_label])
        pos_idx = random.choice([i for i in self.class_indices[anchor_label] if i != anchor_idx])
        neg_label = random.choice([l for l in self.class_indices.keys() if l != anchor_label])
        neg_idx = random.choice(self.class_indices[neg_label])
        return anchor_idx, pos_idx, neg_idx

# Transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

train_ds = FaceDataset(DATA_DIR, transform=transform)
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0)

val_ds = FaceDataset(VAL_DIR, transform=transform)
val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=0)
print(val_ds.classes)
# Model and optimizer (MobileNetV2 + ArcFace)
from torchvision import models
import torch.nn.functional as F

class MobileFace(torch.nn.Module):
    def __init__(self, emb_size=512):
        super().__init__()
        backbone = models.mobilenet_v2(weights='IMAGENET1K_V1')
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
        theta = torch.acos(torch.clamp(cosine, -1+1e-7, 1-1e-7))
        target_logit = torch.cos(theta + self.margin)
        one_hot = F.one_hot(labels, num_classes=cosine.size(1)).float()
        logits = cosine * (1 - one_hot) + target_logit * one_hot
        return logits * self.scale

# Model and optimizer
NUM_CLASSES = len(train_ds.classes)
model = MobileFace(emb_size=512).to(DEVICE)
arcface = ArcFaceHead(emb_size=512, num_classes=NUM_CLASSES, scale=64.0, margin=MARGIN).to(DEVICE)
optimizer = torch.optim.Adam(list(model.parameters()) + list(arcface.parameters()), lr=LR, weight_decay=WEIGHT_DECAY)
criterion = torch.nn.CrossEntropyLoss()

emissions_output_path = "reports/emissions.csv"
Path("reports").mkdir(parents=True, exist_ok=True)


def train_model():
    mlflow.log_params({'batch_size': BATCH, 'epochs': EPOCHS, 'lr': LR, 'margin': MARGIN, 'weight_decay': WEIGHT_DECAY})

    print("\nStarting CodeCarbon Emissions Tracker.")

    # Initialize the tracker with a project name for better organization
    # Context manager ensures the tracker stops automatically
    with EmissionsTracker(project_name="Face_embedding", output_file=emissions_output_path) as tracker:

        for epoch in range(EPOCHS):
            with mlflow.start_run(nested=True, run_name=f"epoch_{epoch}") as epoch_run:
                model.train()
                total_loss = 0.0

                for images, labels in tqdm(train_loader, desc=f'Epoch {epoch}'):
                    images, labels = images.to(DEVICE), labels.to(DEVICE)

                    emb = model(images)
                    logits = arcface(emb, labels)  # apply ArcFace margin
                    loss = criterion(logits, labels)  # CE loss with margin-adjusted logits

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                avg_loss = total_loss / max(1, len(train_loader))
                mlflow.log_metric('train_loss', avg_loss, step=epoch)
                print(f'Epoch {epoch} avg loss {avg_loss:.4f}')

                # Validation
                model.eval()
                val_loss = 0.0
                correct = 0
                total = 0
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
                val_loss /= max(1, len(val_loader))
                val_acc = correct / max(1, total)
                mlflow.log_metric('val_loss', val_loss, step=epoch)
                mlflow.log_metric('val_accuracy', val_acc, step=epoch)
                print(f'Validation loss {val_loss:.4f}, acc {val_acc:.4f}')
            
        # Log the final emissions to MLflow
        if Path(emissions_output_path).exists(): 
            # Read the last entry from emissions.csv to get all the details
            try:
                df = pd.read_csv(emissions_output_path).tail(1) 
                
                # Log the key environmental metrics
                mlflow.log_metric('carbon_emissions_kg_co2', df['emissions'].iloc[0])
                mlflow.log_metric('energy_consumed_kwh', df['energy_consumed'].iloc[0])
                mlflow.log_param('compute_location', f"{df['country_name'].iloc[0]} ({df['region'].iloc[0]})")
                
                print(f"CodeCarbon successfully logged {df['emissions'].iloc[0]:.6f} kg CO₂ to MLflow.")

            except Exception as e:
                print(f"Error logging CodeCarbon metrics to MLflow: {e}")

    # Save model
    model_dir = Path('models/face_embedding')
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / 'mobilenetv2_arcface_epoch_last.pt'
    torch.save(model.state_dict(), model_path)
    mlflow.log_artifact(str(model_path))
    mlflow.pytorch.log_model(model, 'pytorch_model')
    print(f'Training finished — model saved to {model_dir}')


if __name__ == "__main__":
    # Get experiment id or create one
    experiment_id = params['mlflow'].get('experiment_id')
    if not experiment_id:
        experiment_name = params['mlflow'].get('experiment_name', 'face_embedding')
        mlflow.set_experiment(experiment_name)
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id
        params['mlflow']['experiment_id'] = experiment_id
        with open("params.yaml", "w") as f:
            yaml.safe_dump(params, f)
    else:
        mlflow.set_experiment(params['mlflow'].get('experiment_name', 'face_embedding'))

    # Start top-level run with experiment_id
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as parent:
        params['mlflow']['run_id'] = parent.info.run_id
        with open("params.yaml", "w") as f:
            yaml.safe_dump(params, f)
        # Nested run for training
        with mlflow.start_run(nested=True, run_name="train_model"):
            train_model()