#!/usr/bin/env python3
"""
Train a lightweight face real/fake classifier.

Outputs:
  models/fake_classification/
      model_best.pth
      model_last.pth
      model_scripted.pt
      metrics_history.csv
      metrics_val.json
      config.json
  reports/fake_classification/
      emissions.csv
"""

from __future__ import annotations
import argparse, json, logging, time, os
from pathlib import Path
from typing import Tuple, Dict
import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
import yaml
import pandas as pd
from dotenv import load_dotenv

# --- Shared utilities ---
from train_util import (
    init_mlflow, start_emissions_tracker, get_device,
    log_metrics_mlflow, log_params_mlflow, prepare_output_dirs
)

# -------------------- Load params --------------------
splits_dir = Path("data/processed/fake_classification/split")
norm_dir = Path("data/fake_classification/preprocessing/val/normalization.json")
out_dir = Path("models/fake_classification")

with open("pipelines/fake_classification/params.yaml", encoding="utf-8") as f:
    params = yaml.safe_load(f)

# Hyperparameters
backbone = params["training"]["backbone"]
batch_size = params["training"]["batch_size"]
epochs = params["training"]["epochs"]
freeze_backbone = params["training"]["freeze_backbone"]
lr = params["training"]["lr"]
weight_decay = params["training"]["weight_decay"]
num_workers = params["training"]["num_workers"]
export_onnx = params["training"]["export_onnx"]

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

device_param = params['training']['device'].lower()
if device_param == 'cuda' and torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif device_param == 'mps' and getattr(torch.backends, 'mps', None) is not None:
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')
print("Using device:", DEVICE)


# -------------------- Helper utils --------------------
def setup_logger(v: int = 1):
    level = logging.WARNING if v == 0 else (logging.INFO if v == 1 else logging.DEBUG)
    logging.basicConfig(level=level,
                        format='[%(asctime)s] %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')


def load_norm(norm_json: Path | None) -> Tuple[list[float], list[float]]:
    if not norm_json or not Path(norm_json).exists():
        return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    obj = json.loads(Path(norm_json).read_text())
    mean = obj.get("mean", [0.485, 0.456, 0.406])
    std = obj.get("std", [0.229, 0.224, 0.225])
    if len(mean) == 1: mean = mean * 3
    if len(std) == 1: std = std * 3
    return mean, std


def build_model(backbone: str, num_classes: int,
                pretrained: bool = True, freeze_backbone: bool = False) -> nn.Module:
    if backbone == "mobilenet_v3_small":
        weights = torchvision.models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        m = torchvision.models.mobilenet_v3_small(weights=weights)
        in_feat = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_feat, num_classes)
        head_names = ["classifier.3", "classifier.1"]
    elif backbone == "resnet18":
        weights = torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
        m = torchvision.models.resnet18(weights=weights)
        in_feat = m.fc.in_features
        m.fc = nn.Linear(in_feat, num_classes)
        head_names = ["fc"]
    elif backbone == "efficientnet_b0":
        weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        m = torchvision.models.efficientnet_b0(weights=weights)
        in_feat = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_feat, num_classes)
        head_names = ["classifier.1", "classifier.3"]
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    if freeze_backbone:
        for name, p in m.named_parameters():
            p.requires_grad = any(k in name for k in head_names)
    return m


def make_dataloaders(splits_dir: Path, mean, std, batch_size: int, workers: int, size=224):
    tfm = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train_ds = datasets.ImageFolder(Path(splits_dir) / "train", transform=tfm)
    val_ds = datasets.ImageFolder(Path(splits_dir) / "val", transform=tfm)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=workers, pin_memory=True)
    return train_loader, val_loader


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Evaluating", leave=False):
            x = x.to(device)
            logits = model(x)
            pred = torch.argmax(logits, dim=1).cpu().numpy().tolist()
            y_pred.extend(pred)
            y_true.extend(y.numpy().tolist())
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    return {"accuracy": float(acc), "precision": float(prec), "recall": float(rec), "f1": float(f1)}

# -------------------- Training --------------------
def main(run_id=None):
    ap = argparse.ArgumentParser(description="Train AI face classifier (real vs fake)")
    ap.add_argument("--splits-dir", type=Path, default=splits_dir)
    ap.add_argument("--norm-json", type=Path, default=norm_dir)
    ap.add_argument("--backbone", choices=["mobilenet_v3_small", "resnet18", "efficientnet_b0"], default=backbone)
    ap.add_argument("--epochs", type=int, default=epochs)
    ap.add_argument("--batch-size", type=int, default=batch_size)
    ap.add_argument("--lr", type=float, default=lr)
    ap.add_argument("--weight-decay", type=float, default=weight_decay)
    ap.add_argument("--num-workers", type=int, default=num_workers)
    ap.add_argument("--freeze-backbone", type=int, default=freeze_backbone)
    ap.add_argument("--export-onnx", type=int, default=export_onnx)
    ap.add_argument("--out-dir", type=Path, default=out_dir)
    ap.add_argument("-v", "--verbose", action="count", default=1)
    args = ap.parse_args()

    setup_logger(args.verbose)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    mean, std = load_norm(args.norm_json)
    train_loader, val_loader = make_dataloaders(args.splits_dir, mean, std, args.batch_size, args.num_workers)
    classes = train_loader.dataset.classes
    label_map = {i: c for i, c in enumerate(classes)}

    model = build_model(args.backbone, num_classes=len(classes),
                        pretrained=True, freeze_backbone=bool(args.freeze_backbone)).to(DEVICE)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    # --- Setup MLflow + outputs ---
    model_dir, reports_dir = prepare_output_dirs("fake_classification")
    log_params_mlflow({
        "backbone": args.backbone,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "num_workers": args.num_workers,
        "freeze_backbone": bool(args.freeze_backbone),
        "device": str(DEVICE),
        "num_classes": len(classes)
    })

    metrics_csv = args.out_dir / "metrics_history.csv"
    if not metrics_csv.exists():
        metrics_csv.write_text("epoch,train_loss,accuracy,precision,recall,f1\n")

    with start_emissions_tracker("fake_classification", reports_dir):
        best_f1, best_val_acc, best_val_loss, best_epoch = -1.0, 0.0, float("inf"), 0
        for epoch in range(1, args.epochs + 1):
            with torch.set_grad_enabled(True):
                model.train()
                total_loss = 0.0
                for x, y in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    optimizer.zero_grad()
                    logits = model(x)
                    loss = criterion(logits, y)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item() * x.size(0)

                avg_train_loss = total_loss / len(train_loader.dataset)
                metrics_val = evaluate(model, val_loader, DEVICE)
                log_metrics_mlflow({
                    "train_loss": avg_train_loss,
                    "val_accuracy": metrics_val["accuracy"],
                    "val_precision": metrics_val["precision"],
                    "val_recall": metrics_val["recall"],
                    "val_f1": metrics_val["f1"]
                }, step=epoch)

                with metrics_csv.open("a", encoding="utf-8") as f:
                    f.write(f"{epoch},{avg_train_loss:.6f},{metrics_val['accuracy']:.6f},"
                            f"{metrics_val['precision']:.6f},{metrics_val['recall']:.6f},{metrics_val['f1']:.6f}\n")

                if metrics_val["f1"] > best_f1:
                    best_f1 = metrics_val["f1"]
                    best_val_acc = metrics_val["accuracy"]
                    best_val_loss = avg_train_loss
                    best_epoch = epoch
                    torch.save({"state_dict": model.state_dict(),
                                "backbone": args.backbone,
                                "classes": classes}, args.out_dir / "model_best.pth")
                    (args.out_dir / "label_map.json").write_text(json.dumps(label_map, indent=2))
                    (args.out_dir / "metrics_val.json").write_text(json.dumps(metrics_val, indent=2))

    # Save last model and config
    torch.save({"state_dict": model.state_dict(),
                "backbone": args.backbone, "classes": classes},
               args.out_dir / "model_last.pth")
    (args.out_dir / "config.json").write_text(json.dumps({
        "splits_dir": str(args.splits_dir),
        "norm_json": str(args.norm_json),
        "backbone": args.backbone,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "num_workers": args.num_workers,
        "freeze_backbone": bool(args.freeze_backbone),
        "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }, indent=2))

    # --- Log artifacts + metrics ---
    for artifact in ["metrics_history.csv", "metrics_val.json", "config.json", "label_map.json"]:
        mlflow.log_artifact(args.out_dir / artifact)

    if args.export_onnx and (args.out_dir / "model.onnx").exists():
        mlflow.log_artifact(args.out_dir / "model.onnx")
    if (args.out_dir / "model_scripted.pt").exists():
        mlflow.log_artifact(args.out_dir / "model_scripted.pt")

    # --- Log emissions ---
    try:
        df = pd.read_csv(reports_dir / "emissions.csv").tail(1)
        mlflow.log_metric("carbon_emissions_kg", df["emissions"].iloc[0])
        mlflow.log_metric("energy_kwh", df["energy_consumed"].iloc[0])
        mlflow.set_tags({
            "gpu_model": df.get("gpu_model", ["unknown"]).iloc[0],
            "cpu_model": df.get("cpu_model", ["unknown"]).iloc[0],
            "compute_region": f"{df['country_name'].iloc[0]} ({df['region'].iloc[0]})"
        })
    except Exception as e:
        print(f"⚠️ Could not log emissions: {e}")

    mlflow.log_metrics({
        "best_f1": best_f1,
        "best_val_accuracy": best_val_acc,
        "best_val_loss": best_val_loss
    })
    mlflow.set_tags({
        "project": "fake_classification",
        "framework": "pytorch",
        "architecture": args.backbone,
        "experiment_type": "fine_tune" if args.freeze_backbone else "baseline"
    })

    print(f"✅ Training complete — best F1: {best_f1:.4f}, model saved at {args.out_dir.resolve()}")

if __name__ == "__main__":
        # Get experiment id or create one
    experiment_id = params['mlflow'].get('experiment_id')
    if not experiment_id:
        experiment_name = params['mlflow'].get('experiment_name', 'face_embedding')
        mlflow.set_experiment(experiment_name)
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id
        params['mlflow']['experiment_id'] = experiment_id
        with open("pipelines/fake_classification/params.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(params, f)
    else:
        mlflow.set_experiment(params['mlflow'].get('experiment_name', 'face_embedding'))


    # Start top-level run with experiment_id
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as parent:
        params['mlflow']['run_id'] = parent.info.run_id
        with open("pipelines/fake_classification/params.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(params, f)

        # Nested run for training
        with mlflow.start_run(nested=True, run_name="train_model"):
            main(run_id=mlflow.active_run().info.run_id)