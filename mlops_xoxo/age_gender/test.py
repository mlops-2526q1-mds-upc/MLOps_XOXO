"""
test.py - Final test on test set with MLflow tracking
"""

import pandas as pd
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import json
import numpy as np
from dotenv import load_dotenv
# MLflow imports
import mlflow
import yaml
import os

# Load parameters
with open('pipelines/age_gender/params.yaml', 'r') as f:
    params = yaml.safe_load(f)

# Configuration
DATA_DIR = Path(params['dataset']['raw_dir'])
TEST_FILE = Path(params['dataset']['processed_dir']) / "splits/test.csv"
MODELS_DIR = Path("models/age_gender")
OUTPUT_DIR = Path("reports/age_gender/test_results")
# -------------------- Environment --------------------
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

# Set experiment
mlflow.set_experiment(params['mlflow']['experiment_name'])
device_param = params['training']['device'].lower()
if device_param == 'cuda' and torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif device_param == 'mps' and getattr(torch.backends, 'mps', None) is not None:
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')
print("Using device:", DEVICE)

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
    
    task = 'gender' if 'gender' in model_path.name else 'age'
    num_outputs = 2 if task == 'gender' else 1
    
    model = SimpleModel(num_outputs)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    
    return model, task


def test(model, loader, task):
    """Test model"""
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
        
        # Confusion matrix
        cm = [[0, 0], [0, 0]]
        for p, l in zip(all_preds, all_labels):
            cm[int(l)][int(p)] += 1
        
        return {
            'accuracy': float(accuracy),
            'correct': int(correct),
            'total': int(len(all_labels)),
            'confusion_matrix': [[int(cm[0][0]), int(cm[0][1])],
                                [int(cm[1][0]), int(cm[1][1])]]
        }
    else:
        errors = [abs(p - l) for p, l in zip(all_preds, all_labels)]
        mae = sum(errors) / len(errors)
        rmse = (sum(e**2 for e in errors) / len(errors)) ** 0.5
        
        # Accuracy within thresholds
        within_5 = sum(1 for e in errors if e <= 5) / len(errors) * 100
        within_10 = sum(1 for e in errors if e <= 10) / len(errors) * 100
        
        return {
            'mae': float(mae),
            'rmse': float(rmse),
            'within_5_years': float(within_5),
            'within_10_years': float(within_10),
            'total': int(len(all_labels))
        }


def main():
    print("=" * 60)
    print("FINAL TEST")
    print("=" * 60)
    
    # Load test data
    from train import UTKFaceDataset, get_transforms
    test_df = pd.read_csv(TEST_FILE)
    print(f"\nTest set: {len(test_df)} images")
    
    results = {}
    
    # Test each model (create MLflow runs)
    for model_path in MODELS_DIR.glob("*_model.pt"):
        print(f"\n{'='*60}")
        print(f"Testing: {model_path.name}")
        print(f"{'='*60}")
        
        model, task = load_model(model_path)
        
        dataset = UTKFaceDataset(test_df, DATA_DIR, get_transforms(), task)
        loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
        
        # Start MLflow run for testing
        with mlflow.start_run(run_name=f"{task}_test") as run:
            
            # Log tags
            mlflow.set_tag("stage", "test")
            mlflow.set_tag("model_type", task)
            
            # Test
            metrics = test(model, loader, task)
            results[task] = metrics
            
            # Log metrics to MLflow
            if task == 'gender':
                mlflow.log_metric("test_accuracy", metrics['accuracy'])
                mlflow.log_metric("test_correct", metrics['correct'])
                print(f"Accuracy: {metrics['accuracy']*100:.2f}%")
                print(f"Confusion Matrix:")
                print(f"  [[{metrics['confusion_matrix'][0][0]}, {metrics['confusion_matrix'][0][1]}],")
                print(f"   [{metrics['confusion_matrix'][1][0]}, {metrics['confusion_matrix'][1][1]}]]")
            else:
                mlflow.log_metric("test_mae", metrics['mae'])
                mlflow.log_metric("test_rmse", metrics['rmse'])
                mlflow.log_metric("test_within_5_years", metrics['within_5_years'])
                mlflow.log_metric("test_within_10_years", metrics['within_10_years'])
                print(f"MAE: {metrics['mae']:.2f} years")
                print(f"RMSE: {metrics['rmse']:.2f} years")
                print(f"Within ±5 years: {metrics['within_5_years']:.1f}%")
                print(f"Within ±10 years: {metrics['within_10_years']:.1f}%")
            
            print(f"✅ MLflow Run ID: {run.info.run_id}")
    
    # Save results (with type conversion)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "test_results.json"
    
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