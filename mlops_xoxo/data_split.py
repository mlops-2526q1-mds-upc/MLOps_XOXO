# data_split.py
import os
import shutil
import json
from pathlib import Path
from utils.data_utils import create_manifest

# Paths
RAW_DIR = Path("data/raw")              # raw images by person
MANIFEST_PATH = Path("data/processed/splits/manifest.json")  # manifest with splits
OUTPUT_DIR = Path("data/processed")              # root output folder

def manifest(RAW_DIR=RAW_DIR, MANIFEST_PATH=MANIFEST_PATH):
    # Step 2: Create train/val/test manifest
    create_manifest(RAW_DIR, MANIFEST_PATH)

SPLITS = ["train", "val", "test"]

def split_data(raw_dir, manifest_path, output_dir):
    with open(manifest_path) as f:
        manifest = json.load(f)

    for split in SPLITS:
        split_dir = output_dir / split
        for img_path in manifest.get(split, []):
            img_path = Path(img_path)
            person_id = img_path.parent.name
            target_dir = split_dir / person_id
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(img_path, target_dir / img_path.name)
        print(f"{split}: copied {len(manifest.get(split, []))} images")

if __name__ == "__main__":
    manifest(RAW_DIR, MANIFEST_PATH)
    split_data(RAW_DIR, MANIFEST_PATH, OUTPUT_DIR)
    print("Data split completed!")