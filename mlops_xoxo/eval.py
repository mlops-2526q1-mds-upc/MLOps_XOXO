# eval.py
import yaml
import torch
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms
from pathlib import Path
from PIL import Image
import numpy as np
from collections import defaultdict
import json
import mlflow
from dotenv import load_dotenv
import os
from utils.mlflow_run_decorator import mlflow_run
# Load params
with open("params.yaml") as f:
    params = yaml.safe_load(f)

# Device setup
DEVICE = torch.device('mps') if getattr(torch.backends, 'mps', None) else torch.device('cpu')
print("Using device:", DEVICE)

# Load model
model_path = Path('models/face_embedding/facenet_epoch_last.pt')
model = InceptionResnetV1(pretrained=None, classify=False).to(DEVICE)
state_dict = torch.load(model_path, map_location=DEVICE)
model.load_state_dict(state_dict, strict=False)  # ignore extra logits keys
model.eval()

# Image transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

# Load manifest
OUT = Path(params['dataset']['processed_dir'])
manifest = json.load(open(OUT /'splits'/ 'manifest.json'))

@mlflow_run
def evaluate_model():
    # Build gallery: mean embedding per identity from train
    gallery = {}
    embs_by_id = defaultdict(list)
    for img_path in manifest['train']:
        id_ = Path(img_path).parent.name
        img = Image.open(img_path).convert('RGB')
        t = transform(img).unsqueeze(0).to(DEVICE)
        e = model(t).detach().cpu().numpy()[0]
        embs_by_id[id_].append(e)

    for k, v in embs_by_id.items():
        gallery[k] = np.mean(v, axis=0)

    # Evaluate top-1 nearest neighbor accuracy on test set
    correct, total = 0, 0
    predictions = []
    for img_path in manifest['test']:
        true_id = Path(img_path).parent.name
        img = Image.open(img_path).convert('RGB')
        t = transform(img).unsqueeze(0).to(DEVICE)
        e = model(t).detach().cpu().numpy()[0]
        sims = {k: e.dot(v) / (np.linalg.norm(e)*np.linalg.norm(v)) for k,v in gallery.items()}
        pred = max(sims, key=sims.get)
        if pred == true_id:
            correct += 1
        total += 1
        predictions.append({
            'image_path': img_path,
            'true_id': true_id,
            'predicted_id': pred
        })

    acc = correct / total if total else 0
    print('Top-1 NN accuracy:', acc)

    # Log to MLflow
    mlflow.log_metric('top1_accuracy', acc)
    mlflow.log_param('eval_total', total)
    mlflow.log_param('model_path', str(model_path))
    
    # Save artifacts
    reports_dir = Path('reports')
    reports_dir.mkdir(exist_ok=True)
    summary_file = reports_dir / 'eval_summary.txt'
    with open(summary_file, 'w') as f:
        f.write(f'top1_accuracy: {acc}\n')
        f.write(f'eval_total: {total}\n')
    mlflow.log_artifact(str(summary_file))

    predictions_file = reports_dir / 'eval_predictions.json'
    with open(predictions_file, 'w') as f:
        json.dump(predictions, f, indent=2)
    mlflow.log_artifact(str(predictions_file))

if __name__ == "__main__":
    evaluate_model()