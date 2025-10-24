#!/usr/bin/env python3
"""
Train a lightweight face real/fake classifier.

Inputs
  --splits-dir : folder with train/ and val/ produced by data_split.py
  --norm-json  : normalization.json from preprocessing (optional; falls back to ImageNet stats)

Outputs (into --out-dir = artifacts/ai_face by default)
  model_best.pth         # best by F1
  model_last.pth         # last epoch
  model.onnx             # optional: when --export-onnx 1
  model_scripted.pt      # TorchScript (portable .pt)
  metrics_history.csv    # per-epoch metrics
  metrics_val.json       # best-epoch snapshot
  label_map.json         # index -> class
  config.json            # hyperparameters and paths
"""

from __future__ import annotations
import argparse, json, logging, time
from pathlib import Path
from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


# -------------------- Utils --------------------
def setup_logger(v: int = 1):
    level = logging.WARNING if v == 0 else (logging.INFO if v == 1 else logging.DEBUG)
    logging.basicConfig(level=level,
                        format='[%(asctime)s] %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')


def load_norm(norm_json: Path | None) -> Tuple[list[float], list[float]]:
    if not norm_json or not Path(norm_json).exists():
        # ImageNet defaults
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
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            pred = torch.argmax(logits, dim=1).cpu().numpy().tolist()
            y_pred.extend(pred)
            y_true.extend(y.numpy().tolist())
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return {"accuracy": float(acc), "precision": float(prec),
            "recall": float(rec), "f1": float(f1)}


# -------------------- Main --------------------
def main():
    ap = argparse.ArgumentParser(description="Train AI face classifier (real vs fake)")
    ap.add_argument("--splits-dir", type=Path, required=True)
    ap.add_argument("--norm-json", type=Path, default=None)
    ap.add_argument("--backbone", choices=["mobilenet_v3_small", "resnet18", "efficientnet_b0"],
                    default="mobilenet_v3_small")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--freeze-backbone", type=int, default=0)
    ap.add_argument("--export-onnx", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=Path("artifacts/ai_face"))
    ap.add_argument("-v", "--verbose", action="count", default=1)
    args = ap.parse_args()

    setup_logger(args.verbose)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    mean, std = load_norm(args.norm_json)
    train_loader, val_loader = make_dataloaders(
        args.splits_dir, mean, std, args.batch_size, args.num_workers
    )
    classes = train_loader.dataset.classes
    label_map = {i: c for i, c in enumerate(classes)}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.backbone, num_classes=len(classes),
                        pretrained=True, freeze_backbone=bool(args.freeze_backbone)).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    # prepare metrics CSV
    metrics_csv = args.out_dir / "metrics_history.csv"
    if not metrics_csv.exists():
        metrics_csv.write_text("epoch,train_loss,accuracy,precision,recall,f1\n")

    best_f1 = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running += loss.item() * x.size(0)

        train_loss = running / len(train_loader.dataset)
        metrics_val = evaluate(model, val_loader, device)

        # log console + CSV
        logging.info("Epoch %d | loss=%.4f | acc=%.4f | f1=%.4f",
                     epoch, train_loss, metrics_val["accuracy"], metrics_val["f1"])
        with metrics_csv.open("a", encoding="utf-8") as f:
            f.write(f"{epoch},{train_loss:.6f},{metrics_val['accuracy']:.6f},"
                    f"{metrics_val['precision']:.6f},{metrics_val['recall']:.6f},{metrics_val['f1']:.6f}\n")

        # save best by F1
        if metrics_val["f1"] > best_f1:
            best_f1 = metrics_val["f1"]
            torch.save({"state_dict": model.state_dict(),
                        "backbone": args.backbone,
                        "classes": classes},
                       args.out_dir / "model_best.pth")
            (args.out_dir / "label_map.json").write_text(json.dumps(label_map, indent=2))
            (args.out_dir / "metrics_val.json").write_text(json.dumps(metrics_val, indent=2))

    # final checkpoint + config
    torch.save({"state_dict": model.state_dict(),
                "backbone": args.backbone,
                "classes": classes},
               args.out_dir / "model_last.pth")

    (args.out_dir / "config.json").write_text(json.dumps({
        "splits_dir": str(args.splits_dir),
        "norm_json": str(args.norm_json) if args.norm_json else None,
        "backbone": args.backbone,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "num_workers": args.num_workers,
        "freeze_backbone": bool(args.freeze_backbone),
        "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }, indent=2))

    # Optional exports for portability
    if args.export_onnx:
        model.eval()
        dummy = torch.randn(1, 3, 224, 224, device=device)
        torch.onnx.export(model, dummy, args.out_dir / "model.onnx",
                          input_names=["input"], output_names=["logits"],
                          dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
                          opset_version=12)
        logging.info("Exported ONNX to %s", (args.out_dir / "model.onnx").resolve())

    try:
        model_cpu = model.to("cpu").eval()
        scripted = torch.jit.script(model_cpu)
        scripted.save(str(args.out_dir / "model_scripted.pt"))
        logging.info("Saved TorchScript to %s", (args.out_dir / "model_scripted.pt").resolve())
    except Exception as e:
        logging.warning("TorchScript export failed: %s", e)

    logging.info("Training complete. Best F1=%.4f", best_f1)
    print(str(args.out_dir.resolve()))


if __name__ == "__main__":
    main()
