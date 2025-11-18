"""
eval.py - Evaluate models on validation set with MLflow tracking
"""

import pandas as pd
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import json
import numpy as np
from dotenv import load_dotenv
import os
# MLflow imports
import mlflow
import yaml

# Load parameters
with open('pipelines/age_gender/params.yaml', 'r') as f:
    params = yaml.safe_load(f)

# Configuration
DATA_DIR = Path(params['dataset']['raw_dir'])
VAL_FILE = Path(params['dataset']['processed_dir']) / "splits/val.csv"
MODELS_DIR = Path("models/age_gender")
OUTPUT_DIR = Path("reports/age_gender/evaluation")
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
run_name = params['mlflow'].get('run_name', 'default_run')
device_param = params['training']['device'].lower()
if device_param == 'cuda' and torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif device_param == 'mps' and getattr(torch.backends, 'mps', None) is not None:
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')
print("Using device:", DEVICE)

# Set experiment
mlflow.set_experiment(params['mlflow']['experiment_name'])


def convert_to_python_types(obj):
    """Convert numpy/torch types to Python native types for JSON serialization"""
    if isinstance(obj, dict):
        return {k: convert_to_python_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_python_types(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (torch.Tensor,)):
        return obj.cpu().numpy().tolist()
    return obj


def load_model(model_path):
    """Load trained model"""
    from train import SimpleModel
    
    # Determine task from filename
    task = 'gender' if 'gender' in model_path.name else 'age'
    num_outputs = 2 if task == 'gender' else 1
    
    model = SimpleModel(num_outputs)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    
    return model, task


def evaluate(model, loader, task):
    """Evaluate model"""
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            outputs = model(images)
            
            if task == 'gender':
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
            else:
                preds = outputs.cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    
    # Calculate metrics
    if task == 'gender':
        correct = sum(p == l for p, l in zip(all_preds, all_labels))
        accuracy = correct / len(all_labels)
        return {
            'accuracy': float(accuracy),
            'correct': int(correct),
            'total': int(len(all_labels))
        }
    else:
        errors = [abs(p - l) for p, l in zip(all_preds, all_labels)]
        mae = sum(errors) / len(errors)
        return {
            'mae': float(mae),
            'total': int(len(all_labels))
        }


def main():
    print("=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    
    # Load validation data
    from train import UTKFaceDataset, get_transforms
    val_df = pd.read_csv(VAL_FILE)
    print(f"\nValidation set: {len(val_df)} images")
    
    results = {}
    
    # Evaluate each model (create MLflow runs)
    for model_path in MODELS_DIR.glob("*_model.pt"):
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_path.name}")
        print(f"{'='*60}")
        
        model, task = load_model(model_path)
        
        dataset = UTKFaceDataset(val_df, DATA_DIR, get_transforms(), task)
        loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
        
        # Start MLflow run for evaluation
        with mlflow.start_run(run_name=f"{task}_evaluation") as run:
            
            # Log tags
            mlflow.set_tag("stage", "evaluation")
            mlflow.set_tag("model_type", task)
            
            # Evaluate
            metrics = evaluate(model, loader, task)
            results[task] = metrics
            
            # Log metrics to MLflow
            if task == 'gender':
                mlflow.log_metric("eval_accuracy", metrics['accuracy'])
                print(f"Accuracy: {metrics['accuracy']*100:.2f}%")
            else:
                mlflow.log_metric("eval_mae", metrics['mae'])
                print(f"MAE: {metrics['mae']:.2f} years")
            
            print(f"✅ MLflow Run ID: {run.info.run_id}")
    
    # Save results (with type conversion)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "eval_results.json"
    
    # Convert all numpy/torch types to Python types
    results_clean = convert_to_python_types(results)
    
    with open(output_path, 'w') as f:
        json.dump(results_clean, f, indent=4)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_path}")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    exit(main())