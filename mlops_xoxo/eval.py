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
from .train import MobileFace

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
DEVICE = torch.device('mps') if getattr(torch.backends, 'mps', None) else torch.device('cpu')
print("Using device:", DEVICE)

# Load model
model_path = Path('models/face_embedding/mobilenetv2_arcface_epoch_last.pt')
model = MobileFace().to(DEVICE)
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
TEST_DIR = OUT / "test"

def calculate_metrics(model, manifest, transform, device):
    # Build gallery
    gallery = {}
    embs_by_id = defaultdict(list)
    for img_path in manifest['train']:
        id_ = Path(img_path).parent.name
        img = Image.open(img_path).convert('RGB')
        t = transform(img).unsqueeze(0).to(device)
        e = model(t).detach().cpu().numpy()[0]
        embs_by_id[id_].append(e)

    for k, v in embs_by_id.items():
        gallery[k] = np.mean(v, axis=0)

    # Evaluate on test set
    correct_top1, correct_top5, total = 0, 0, 0
    predictions = []
    for img_path in manifest['test']:
        true_id = Path(img_path).parent.name
        img = Image.open(img_path).convert('RGB')
        t = transform(img).unsqueeze(0).to(device)
        e = model(t).detach().cpu().numpy()[0]
        sims = {k: e.dot(v) / (np.linalg.norm(e)*np.linalg.norm(v)) for k,v in gallery.items()}
        sorted_ids = sorted(sims, key=sims.get, reverse=True)
        pred_top1 = sorted_ids[0]
        pred_top5 = sorted_ids[:5]
        if pred_top1 == true_id:
            correct_top1 += 1
        if true_id in pred_top5:
            correct_top5 += 1
        total += 1
        predictions.append({
            'image_path': img_path, 'true_id': true_id,
            'predicted_id': pred_top1, 'top5_ids': pred_top5
        })

    # Return all results in a dictionary
    return {
        "acc_top1": correct_top1 / total if total else 0,
        "acc_top5": correct_top5 / total if total else 0,
        "total": total,
        "predictions": predictions
    }

def save_results(results):
    # This function handles logging and file writing
    print('Top-1 NN accuracy:', results['acc_top1'])
    print('Top-5 NN accuracy:', results['acc_top5'])

    # Log to MLflow
    mlflow.log_metric('top1_accuracy', results['acc_top1'])
    mlflow.log_metric('top5_accuracy', results['acc_top5'])
    mlflow.log_param('eval_total', results['total'])
    mlflow.log_param('model_path', str(model_path))
    
    # Save artifacts
    reports_dir = Path('reports')
    reports_dir.mkdir(exist_ok=True)
    summary_file = reports_dir / 'eval_summary.txt'
    with open(summary_file, 'w') as f:
        f.write(f"top1_accuracy: {results['acc_top1']}\n")
        f.write(f"top5_accuracy: {results['acc_top5']}\n")
        f.write(f"eval_total: {results['total']}\n")
    mlflow.log_artifact(str(summary_file))
    
    mlflow.set_tags({
        "model_type": "MobileNetV2+ArcFace",
        "test_dataset": str(TEST_DIR),
        "test_top1_nn_acc": results['acc_top1'],
        "test_top5_nn_acc": results['acc_top5']
    })

    predictions_file = reports_dir / 'eval_predictions.json'
    with open(predictions_file, 'w') as f:
        json.dump(results['predictions'], f, indent=2)
    mlflow.log_artifact(str(predictions_file))

def evaluate_model():
    # This function orchestrates the model evaluation and logging

    results = calculate_metrics(model, manifest, transform, DEVICE)
    save_results(results)


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
    parent_run_id = params['mlflow']['run_id']
    with mlflow.start_run(experiment_id=experiment_id, run_id=parent_run_id) as parent:
        with open("params.yaml", "w") as f:
            yaml.safe_dump(params, f)

        # Nested run for training
        with mlflow.start_run(nested=True, run_name="evaluate_model"):
            evaluate_model()