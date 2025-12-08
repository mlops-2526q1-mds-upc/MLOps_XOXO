#!/usr/bin/env python3
"""Train script for emotion classification using ResNet18 adapted to 1-channel.

This is adapted from an existing training script and integrated with the project's
params and train_util helpers. It reads params from `pipelines/emotion_classification/params.yaml`.
"""
import os
import yaml
import json
import random
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from torchvision.models import resnet18, ResNet18_Weights

import mlflow
from train_util import (
    prepare_output_dirs,
    init_mlflow,
    log_params_mlflow,
    log_metrics_mlflow,
    get_device,
    set_seed,
    start_emissions_tracker,
)


def build_resnet18(num_classes: int, pretrained: bool = True, in_channels: int = 1):
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    if in_channels != 3:
        old_conv = model.conv1
        model.conv1 = nn.Conv2d(in_channels, old_conv.out_channels,
                                kernel_size=old_conv.kernel_size,
                                stride=old_conv.stride,
                                padding=old_conv.padding,
                                bias=(old_conv.bias is not None))
        if pretrained and getattr(old_conv, "weight", None) is not None:
            with torch.no_grad():
                model.conv1.weight[:] = old_conv.weight.mean(dim=1, keepdim=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


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


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.detach().cpu().numpy())
    avg_loss = running_loss / len(loader.dataset)
    acc = float(accuracy_score(all_labels, all_preds))
    macro_f1 = float(f1_score(all_labels, all_preds, average='macro', zero_division=0))
    return avg_loss, acc, macro_f1


def evaluate(model, loader, device, class_names=None):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    return compute_metrics(np.array(all_labels), np.array(all_preds), class_names=class_names)


def main():
    with open("pipelines/emotion_classification/params.yaml", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    seed = params.get('seed', 42)
    set_seed(seed)
    print(f"[INFO] Seed set to {seed}")

    model_dir, report_dir = prepare_output_dirs("emotion_classification")
    print(f"[INFO] Model directory: {model_dir}")
    print(f"[INFO] Report directory: {report_dir}")
    
    # Initialize MLflow
    _, experiment_id = init_mlflow(params)
    experiment_name = params.get('mlflow', {}).get('experiment_name', 'emotion_classification')
    mlflow.set_experiment(experiment_name)
    print(f"[INFO] MLflow experiment: {experiment_name} (ID: {experiment_id})")

    device = get_device()
    print(f"[INFO] Using device: {device}")

    img_size = params.get('preprocessing', {}).get('image_size', 48)
    print(f"[INFO] Image size: {img_size}x{img_size}")
    
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=10, translate=(0.08, 0.08)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    val_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    data_root = Path(params['dataset']['processed_dir']) / 'emotion_classification'
    train_dir = data_root / 'train'
    val_dir = data_root / 'val'

    print(f"[INFO] Loading training data from {train_dir}")
    full_train_dataset = datasets.ImageFolder(str(train_dir), transform=train_transform)
    class_names = full_train_dataset.classes
    num_classes = len(class_names)
    print(f"[INFO] Classes found: {class_names}")
    print(f"[INFO] Number of classes: {num_classes}")

    # stratified val split from train if no separate val present
    if not val_dir.exists():
        print(f"[INFO] Validation directory not found, creating stratified split from training data")
        targets = [y for _, y in full_train_dataset.imgs]
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        train_idx, val_idx = next(sss.split(np.zeros(len(targets)), targets))
        train_subset = Subset(full_train_dataset, train_idx)
        full_train_for_val = datasets.ImageFolder(str(train_dir), transform=val_transform)
        val_subset = Subset(full_train_for_val, val_idx)
        print(f"[INFO] Train subset size: {len(train_subset)}, Val subset size: {len(val_subset)}")
    else:
        print(f"[INFO] Loading validation data from {val_dir}")
        train_subset = full_train_dataset
        val_subset = datasets.ImageFolder(str(val_dir), transform=val_transform)
        print(f"[INFO] Train dataset size: {len(train_subset)}, Val dataset size: {len(val_subset)}")

    batch_size = params.get('training', {}).get('batch_size', 32)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2)
    print(f"[INFO] Batch size: {batch_size}")

    print("[INFO] Building ResNet18 model with 1-channel input...")
    model = build_resnet18(num_classes=num_classes, pretrained=True, in_channels=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=params.get('training', {}).get('lr', 1e-3))
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    epochs = params.get('training', {}).get('epochs', 20)
    best_val_macro_f1 = -1.0
    best_epoch = 0
    history = {'train_loss': [], 'train_acc': [], 'train_macro_f1': [], 'val_macro_f1': [], 'val_acc': []}

    # Start MLflow run
    run_name = params.get('mlflow', {}).get('run_name', 'emotion_train')
    print(f"[INFO] Starting MLflow run: {run_name}")
    with mlflow.start_run(run_name=run_name, experiment_id=experiment_id):
        # Log parameters
        log_params_mlflow({
            'batch_size': batch_size,
            'lr': params.get('training', {}).get('lr', 1e-3),
            'epochs': epochs,
            'num_classes': num_classes,
            'device': str(device),
            'image_size': img_size,
        })
        print("[INFO] Logged training parameters to MLflow")

        print(f"[INFO] Starting training for {epochs} epochs...")
        with start_emissions_tracker('emotion_classification', report_dir):
            for epoch in range(1, epochs + 1):
                train_loss, train_acc, train_macro_f1 = train_one_epoch(model, train_loader, optimizer, criterion, device)
                val_metrics = evaluate(model, val_loader, device, class_names=class_names)
                val_acc = val_metrics['accuracy']
                val_macro_f1 = val_metrics['macro_f1']

                history['train_loss'].append(train_loss)
                history['train_acc'].append(train_acc)
                history['train_macro_f1'].append(train_macro_f1)
                history['val_acc'].append(val_acc)
                history['val_macro_f1'].append(val_macro_f1)

                print(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train Macro F1: {train_macro_f1:.4f}")
                print(f"                 Val Acc: {val_acc:.4f} | Val Macro F1: {val_macro_f1:.4f}")

                scheduler.step(val_macro_f1)

                last_model_path = model_dir / 'last_model.pth'
                torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, last_model_path)

                if val_macro_f1 > best_val_macro_f1:
                    best_val_macro_f1 = val_macro_f1
                    best_epoch = epoch
                    best_model_path = model_dir / 'best_model.pth'
                    torch.save(model.state_dict(), best_model_path)
                    print(f"[INFO] ✓ New best model saved at epoch {epoch} (val_macro_f1={best_val_macro_f1:.4f})")

                # Log metrics to MLflow
                mlflow.log_metrics({
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'train_macro_f1': train_macro_f1,
                    'val_acc': val_acc,
                    'val_macro_f1': val_macro_f1,
                }, step=epoch)

        print("[INFO] Training complete!")
        
        # Final evaluation on val
        best_model_path = model_dir / 'best_model.pth'
        if best_model_path.exists():
            print(f"[INFO] Loading best model from {best_model_path}")
            model.load_state_dict(torch.load(best_model_path, map_location=device))
        test_metrics = evaluate(model, val_loader, device, class_names=class_names)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        test_metrics_path = report_dir / f"test_metrics_{timestamp}.json"
        test_metrics_path.write_text(json.dumps(test_metrics, indent=2))
        print(f"[INFO] Saved test metrics to {test_metrics_path}")
        print(f"[INFO] Final validation accuracy: {test_metrics['accuracy']:.4f}")
        print(f"[INFO] Final validation macro F1: {test_metrics['macro_f1']:.4f}")

        # Log final metrics
        mlflow.log_metrics({
            'final_val_accuracy': test_metrics['accuracy'],
            'final_val_macro_f1': test_metrics['macro_f1'],
            'best_epoch': best_epoch,
            'best_val_macro_f1': best_val_macro_f1,
        })

        # Save model artifacts to MLflow
        print("[INFO] Uploading model and artifacts to MLflow...")
        mlflow.log_artifact(str(best_model_path), artifact_path="model")
        print(f"[INFO] ✓ Model uploaded to MLflow")
        
        mlflow.log_artifact(str(test_metrics_path), artifact_path="metrics")
        print(f"[INFO] ✓ Test metrics uploaded to MLflow")
        
        # attempt ONNX export
        if params.get('training', {}).get('save_onnx', False):
            onnx_path = model_dir / 'best_model.onnx'
            model_cpu = model.to('cpu')
            model_cpu.eval()
            dummy = torch.randn(1, 1, img_size, img_size)
            try:
                torch.onnx.export(model_cpu, dummy, str(onnx_path), input_names=['input'], output_names=['output'], opset_version=11)
                print(f"[INFO] Exported ONNX model to {onnx_path}")
                mlflow.log_artifact(str(onnx_path), artifact_path="model")
                print(f"[INFO] ✓ ONNX model uploaded to MLflow")
            except Exception as e:
                print(f"[WARN] ONNX export failed: {e}")

        print("[INFO] ✅ Training complete and all artifacts logged to MLflow!")
        print(f"[INFO] Best model achieved val_macro_f1={best_val_macro_f1:.4f} at epoch {best_epoch}")


if __name__ == "__main__":
    main()
