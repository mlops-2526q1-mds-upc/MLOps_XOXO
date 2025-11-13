#!/usr/bin/env python3
"""
Évaluation du modèle GenderAge
Calcule la précision du genre et l'erreur MAE sur l'âge
"""
import torch, numpy as np, pandas as pd
from torch.utils.data import DataLoader
from pathlib import Path
import cv2
from train_genderage import GenderAgeDataset
from models.genderage_model import GenderAgeModel

def evaluate(model_path, data_dir, device='cpu'):
    model = GenderAgeModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    ds = GenderAgeDataset(Path(data_dir)/'labels.csv', Path(data_dir)/'images')
    loader = DataLoader(ds, batch_size=32, shuffle=False)
    correct, total, age_err = 0, 0, []
    with torch.no_grad():
        for imgs, genders, ages in loader:
            imgs, genders, ages = imgs.to(device), genders.to(device), ages.to(device)
            out = model(imgs)
            pred_gender = out[:, :2].argmax(1)
            correct += (pred_gender == genders).sum().item()
            total += len(genders)
            age_err.extend((out[:, 2]*100 - ages*100).abs().cpu().numpy())
    print(f"\n🎯 Résultats de l'évaluation")
    print(f"  - Précision Genre: {correct/total:.2%}")
    print(f"  - MAE Âge: {np.mean(age_err):.2f} ans")

if __name__ == "__main__":
    evaluate('trained_models/best_genderage_native.pth', 'data/gender_age/val')