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

def main():
    ap = argparse.ArgumentParser(description="Evaluate classifier on a split")
    ap.add_argument("--splits-dir", type=Path, required=True)
    ap.add_argument("--split", choices=["train","val","test"], default="test")
    ap.add_argument("--norm-json", type=Path, default=None)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("artifacts/ai_face_eval"))
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

if __name__ == "__main__":
    main()
