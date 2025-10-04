# train.py
import os
from dotenv import load_dotenv
import yaml
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1
from pathlib import Path
from PIL import Image
import random
from tqdm import tqdm
import mlflow
import mlflow.pytorch
from utils.mlflow_run_decorator import mlflow_run
import pandas as pd
from codecarbon import EmissionsTracker

# Load environment variables
load_dotenv()  # .env is loaded
mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")  

# Load parameters
with open('params.yaml') as f:
    params = yaml.safe_load(f)

experiment_name = params['mlflow']['experiment_name']
# MLflow setup
if mlflow_uri:
    mlflow.set_tracking_uri(mlflow_uri)
mlflow.set_experiment(experiment_name)

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

# Model and optimizer
model = InceptionResnetV1(pretrained='vggface2', classify=False, device=DEVICE).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
criterion = torch.nn.TripletMarginLoss(margin=MARGIN)

emissions_output_path = "reports/emissions.csv"
Path("reports").mkdir(parents=True, exist_ok=True)

@mlflow_run
def train_model():
    mlflow.log_params({'batch_size': BATCH, 'epochs': EPOCHS, 'lr': LR, 'margin': MARGIN, 'weight_decay': WEIGHT_DECAY})

    print("\nStarting CodeCarbon Emissions Tracker.")

    # Initialize the tracker with a project name for better organization
    # Context manager ensures the tracker stops automatically
    with EmissionsTracker(project_name=experiment_name, output_file=emissions_output_path) as tracker:

        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0.0

            for _ in tqdm(train_loader, desc=f'Epoch {epoch}'):
                triplet_indices = [train_ds.sample_triplet() for _ in range(BATCH // 3)]
                if not triplet_indices:
                    continue

                anchors = torch.stack([train_ds[a][0] for a, _, _ in triplet_indices]).to(DEVICE)
                positives = torch.stack([train_ds[p][0] for _, p, _ in triplet_indices]).to(DEVICE)
                negatives = torch.stack([train_ds[n][0] for _, _, n in triplet_indices]).to(DEVICE)

                a_emb, p_emb, n_emb = model(anchors), model(positives), model(negatives)
                loss = criterion(a_emb, p_emb, n_emb)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / max(1, len(train_loader))
            mlflow.log_metric('train_loss', avg_loss, step=epoch)
            print(f'Epoch {epoch} avg loss {avg_loss:.4f}')

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
    model_path = model_dir / 'facenet_epoch_last.pt'
    torch.save(model.state_dict(), model_path)
    mlflow.log_artifact(str(model_path))
    mlflow.pytorch.log_model(model, 'pytorch_model')
    print(f'Training finished — model saved to {model_dir}')

    print(f"MLFLOW_RUN_ID:{mlflow.active_run().info.run_id}")

if __name__ == "__main__":
    train_model()