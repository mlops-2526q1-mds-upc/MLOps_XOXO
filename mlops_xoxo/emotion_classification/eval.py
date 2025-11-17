#!/usr/bin/env python3
"""Evaluate a saved emotion classification model and write a report to MLflow.
"""
import yaml
import json
from pathlib import Path
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import numpy as np
import mlflow


def compute_metrics(y_true, y_pred, class_names=None):
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0).tolist()
    cm = confusion_matrix(y_true, y_pred).tolist()
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    return {
        'accuracy': acc,
        'macro_f1': macro_f1,
        'per_class_f1': per_class_f1,
        'confusion_matrix': cm,
        'report_text': report
    }


def main():
    print("[INFO] Loading evaluation configuration...")
    with open("pipelines/emotion_classification/params.yaml", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    data_root = Path(params['dataset']['processed_dir']) / 'emotion_classification'
    val_dir = data_root / 'val'
    if not val_dir.exists():
        raise FileNotFoundError(f"Val dir not found: {val_dir}")

    print(f"[INFO] Validation data directory: {val_dir}")

    img_size = params.get('preprocessing', {}).get('image_size', 48)
    val_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    print("[INFO] Loading validation dataset...")
    val_ds = datasets.ImageFolder(str(val_dir), transform=val_transform)
    val_loader = DataLoader(val_ds, batch_size=params.get('training', {}).get('batch_size', 32), shuffle=False)
    class_names = val_ds.classes
    print(f"[INFO] Classes: {class_names}")
    print(f"[INFO] Validation set size: {len(val_ds)}")

    model_path = Path(f"models/emotion_classification/best_model.pth")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"[INFO] Loading model from {model_path}...")
    # build model architecture to match saved weights
    from train import build_resnet18
    model = build_resnet18(num_classes=len(class_names), pretrained=False, in_channels=1)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    print("[INFO] ✓ Model loaded successfully")

    print("[INFO] Running evaluation on validation set...")
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(val_loader):
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            if (batch_idx + 1) % 10 == 0:
                print(f"  → Processed {batch_idx + 1}/{len(val_loader)} batches")

    print("[INFO] Computing metrics...")
    metrics = compute_metrics(np.array(all_labels), np.array(all_preds), class_names=class_names)
    
    # Print results to console
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1-Score: {metrics['macro_f1']:.4f}")
    print("\nConfusion Matrix:")
    cm_array = np.array(metrics['confusion_matrix'])
    print(cm_array)
    print("\nClassification Report:")
    print(metrics['report_text'])
    print("="*60 + "\n")

    # Save evaluation artifacts locally
    out_dir = Path("reports/emotion_classification")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eval_summary.txt").write_text(metrics["report_text"])
    (out_dir / "eval_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"[INFO] ✓ Saved evaluation artifacts to {out_dir}")
    
    # Log evaluation results to MLflow
    print("[INFO] Logging evaluation results to MLflow...")
    experiment_name = params.get('mlflow', {}).get('experiment_name', 'emotion_classification')
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name="emotion_eval"):
        mlflow.log_params({
            'batch_size': params.get('training', {}).get('batch_size', 32),
            'image_size': img_size,
            'num_classes': len(class_names),
        })
        
        mlflow.log_metrics({
            'eval_accuracy': metrics['accuracy'],
            'eval_macro_f1': metrics['macro_f1'],
        })
        
        # Log metrics per class
        for i, class_name in enumerate(class_names):
            if i < len(metrics['per_class_f1']):
                mlflow.log_metric(f"eval_{class_name}_f1", metrics['per_class_f1'][i])
        
        # Log artifacts
        mlflow.log_artifact(str(out_dir / "eval_summary.txt"), artifact_path="evaluation")
        mlflow.log_artifact(str(out_dir / "eval_metrics.json"), artifact_path="evaluation")
        
        print(f"[INFO] ✓ Evaluation logged to MLflow")
    
    print("[INFO] ✅ Evaluation complete!")


if __name__ == "__main__":
    main()
