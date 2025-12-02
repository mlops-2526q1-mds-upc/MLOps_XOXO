#!/usr/bin/env python3
"""
Step 5 — Evaluate: load a checkpoint and evaluate on any split (val/test).

Outputs:
  metrics_<split>.json and confusion_matrix_<split>.csv
"""
from __future__ import annotations
import argparse, json, logging
from pathlib import Path
from typing import List
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import pandas as pd
import yaml
from dotenv import load_dotenv
import mlflow
import os
splits_dir = Path("data/processed/fake_classification/split")
norm_dir = Path("data/fake_classification/preprocessing/val/normalization.json")
out_dir = Path("reports/fake_classification")
checkpoint_default = Path("models/fake_classification/model_best.pth")
with open("pipelines/fake_classification/params.yaml", encoding="utf-8") as f:
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

def setup_logger():
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')

def load_norm(norm_json: Path):
    if not norm_json or not Path(norm_json).exists():
        return [0.485,0.456,0.406],[0.229,0.224,0.225]
    obj = json.loads(Path(norm_json).read_text())
    mean, std = obj.get("mean",[0.485,0.456,0.406]), obj.get("std",[0.229,0.224,0.225])
    if len(mean)==1: mean = mean*3
    if len(std)==1: std = std*3
    return mean, std

def build_model(backbone: str, num_classes: int) -> nn.Module:
    if backbone=="mobilenet_v3_small":
        m = torchvision.models.mobilenet_v3_small()
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    elif backbone=="resnet18":
        m = torchvision.models.resnet18()
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif backbone=="efficientnet_b0":
        m = torchvision.models.efficientnet_b0()
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")
    return m

def save_results(results):
    print
    

def main(argv=None):
    ap = argparse.ArgumentParser(description="Evaluate classifier on a split")
    ap.add_argument("--splits-dir", type=Path, default=splits_dir)
    ap.add_argument("--split", choices=["train","val","test"], default="test")
    ap.add_argument("--norm-json", type=Path, default=norm_dir)
    ap.add_argument("--checkpoint", type=Path, default=checkpoint_default)
    ap.add_argument("--out-dir", type=Path, default=out_dir)
    args = ap.parse_args()
    setup_logger()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    mean, std = load_norm(args.norm_json)
    tfm = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize(mean,std)])
    ds = datasets.ImageFolder(Path(args.splits_dir)/args.split, transform=tfm)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    backbone = ckpt.get("backbone","mobilenet_v3_small")
    classes: List[str] = ckpt.get("classes", ds.classes)
    model = build_model(backbone, num_classes=len(classes))
    model.load_state_dict(ckpt["state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.eval(); y_true, y_pred = [], []
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device)
            pred = torch.argmax(model(x), dim=1).cpu().numpy().tolist()
            y_pred.extend(pred); y_true.extend(y.numpy().tolist())

    acc = accuracy_score(y_true,y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true,y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true,y_pred)
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(args.out_dir/f"confusion_matrix_{args.split}.csv")

    metrics = {"split":args.split, "accuracy":float(acc), "precision":float(prec), "recall":float(rec), "f1":float(f1), "classes":classes}
    (args.out_dir/f"metrics_{args.split}.json").write_text(json.dumps(metrics, indent=2))
    logging.info("Eval %s | acc=%.4f f1=%.4f", args.split, acc, f1)
    print(str((args.out_dir/f"metrics_{args.split}.json").resolve()))

    return metrics  # ✅ return to upper function

def evaluate_model():
    """Main orchestrator for model evaluation with MLflow logging."""
    experiment_name = params['mlflow'].get('experiment_name', 'fake_classification')
    mlflow.set_experiment(experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id

    parent_run_id = params['mlflow'].get('run_id')
    if not parent_run_id:
        raise ValueError("⚠️ Missing parent run_id in params.yaml. Run training first before evaluation.")

    # ✅ Start new nested run linked to training
    with mlflow.start_run(experiment_id=experiment_id, run_name="evaluate_model",
                          nested=True, tags={"mlflow.parentRunId": parent_run_id}):
        # Run main evaluation (compute metrics + save JSON/CSV)
        results = main([])

        # Log metrics
        mlflow.log_metrics({
            "eval_accuracy": results["accuracy"],
            "eval_precision": results["precision"],
            "eval_recall": results["recall"],
            "eval_f1": results["f1"]
        })

        # Log artifacts
        split = results["split"]
        metrics_path = out_dir / f"metrics_{split}.json"
        cm_path = out_dir / f"confusion_matrix_{split}.csv"
        if metrics_path.exists():
            mlflow.log_artifact(str(metrics_path))
        if cm_path.exists():
            mlflow.log_artifact(str(cm_path))

        # Add metadata / tags
        mlflow.set_tags({
            "stage": "evaluation",
            "linked_training_run": parent_run_id,
            "split": split,
            "task": "fake_classification",
            "framework": "pytorch",
        })

        print(f"✅ Logged evaluation results to MLflow under run {mlflow.active_run().info.run_id}")

if __name__ == "__main__":
    evaluate_model()
