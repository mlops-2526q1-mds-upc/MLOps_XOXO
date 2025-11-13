#!/usr/bin/env python3
"""
Vérifie que le dataset UTKFace a bien été préparé
"""
import pandas as pd
from pathlib import Path
import yaml
with open("pipelines/age_gender/params.yaml", encoding="utf-8") as f:
    params = yaml.safe_load(f)

dir = params["dataset"]["processed_dir"]
def verify_dataset(output_dir=dir):
    output = Path(output_dir)
    print("\n🔍 Vérification du dataset...")

    checks = []
    for split in ['train', 'val']:
        img_dir = output / split / 'images'
        csv_path = output / split / 'labels.csv'
        img_count = len(list(img_dir.glob('*.jpg'))) if img_dir.exists() else 0
        csv_count = len(pd.read_csv(csv_path)) if csv_path.exists() else 0
        checks.append((split, img_count, csv_count))

    print(f"\n{'Split':<10} {'Images':>10} {'CSV Entries':>15}")
    print("-" * 40)
    for name, img_count, csv_count in checks:
        print(f"{name:<10} {img_count:>10} {csv_count:>15}")

if __name__ == "__main__":
    verify_dataset()