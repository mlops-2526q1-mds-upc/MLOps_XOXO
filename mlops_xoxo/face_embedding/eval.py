# eval.py
import json
import os
from collections import defaultdict
from pathlib import Path
import mlflow
import numpy as np
import torch
import yaml
from dotenv import load_dotenv
from PIL import Image
from torchvision import transforms
from train import MobileFace

with open("pipelines/face_embedding/params.yaml", encoding="utf-8") as f:
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
model_path = Path('models/face_embedding/mobilenetv2_arcface_model.pth')
model = MobileFace().to(DEVICE)
state_dict = torch.load(model_path, map_location=DEVICE)
model.load_state_dict(state_dict, strict=False)  # ignore extra logits keys
model.eval()

# Image transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Load manifest
OUT = Path(params['dataset']['processed_dir'])
with open(OUT / 'splits' / 'manifest.json', "r", encoding="utf-8") as f:
    manifest = json.load(f)
TEST_DIR = OUT / "test"

reports_dir = Path('reports/face_embedding')


def calculate_metrics(model, manifest, transform, device):
    """Function that calculates metrics for model evaluation"""
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
        sims = {k: e.dot(v) / (np.linalg.norm(e)*np.linalg.norm(v)) for k, v in gallery.items()}
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
    """This function handles logging and file writing"""
    print('Top-1 NN accuracy:', results['acc_top1'])
    print('Top-5 NN accuracy:', results['acc_top5'])

    # Log to MLflow
    mlflow.log_metric('top1_accuracy', results['acc_top1'])
    mlflow.log_metric('top5_accuracy', results['acc_top5'])
    mlflow.log_param('eval_total', results['total'])
    mlflow.log_param('model_path', str(model_path))

    # Save artifacts
    reports_dir = Path('reports/face_embedding')
    reports_dir.mkdir(parents=True, exist_ok=True)
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
    with open(predictions_file, 'w', encoding="utf-8") as f:
        json.dump(results['predictions'], f, indent=2)
    mlflow.log_artifact(str(predictions_file))


def evaluate_model():
    """Main orchestrator function for model evaluation"""
    experiment_name = params['mlflow'].get('experiment_name', 'face_embedding')
    mlflow.set_experiment(experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id

    parent_run_id = params['mlflow'].get('run_id')
    if not parent_run_id:
        raise ValueError("⚠️ Missing parent run_id in params.yaml. Run training first before evaluation.")

    # ✅ Create a new nested run under the same parent (even if parent is finished)
    with mlflow.start_run(experiment_id=experiment_id, run_name="evaluate_model",
                          nested=True, tags={"mlflow.parentRunId": parent_run_id}):
        results = calculate_metrics(model, manifest, transform, DEVICE)
        save_results(results)

        mlflow.set_tags({
            "stage": "evaluation",
            "linked_training_run": parent_run_id,
            "task": "face_embedding",
            "framework": "pytorch",
        })

if __name__ == "__main__":
    evaluate_model()
