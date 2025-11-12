#!/usr/bin/env python3
"""Split dataset into train/val/test for emotion classification.

Reads `pipelines/emotion_classification/params.yaml` for directories and ratios.
"""
from pathlib import Path
import argparse
import json
import logging
import random
import shutil
import yaml


with open("pipelines/emotion_classification/params.yaml", encoding="utf-8") as f:
    params = yaml.safe_load(f)

INPUT_DIR = Path(params["dataset"]["raw_dir"]) / "emotion"
OUTPUT_DIR = Path(params["dataset"]["processed_dir"]) / "emotion"
TRAIN_RATIO = params["dataset"]["split"]["train"]
VAL_RATIO = params["dataset"]["split"]["val"]
TEST_RATIO = params["dataset"]["split"]["test"]
SEED = params.get("seed", 42)


def setup_logger():
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")


def split_dataset(in_dir: Path, out_dir: Path, ratios, seed=42, mode="copy"):
    """
    Split nested emotion dataset: data/raw/emotion/{train,test}/{class}/*.jpg
    into data/processed/emotion/{train,val,test}/{class}/*.jpg using provided ratios.
    """
    random.seed(seed)
    
    # Gather all images and their emotion class from train/ and test/ subfolders
    all_images = {}  # {emotion_class: [list of image paths]}
    
    for subset_dir in ["train", "test"]:
        subset_path = in_dir / subset_dir
        if subset_path.exists():
            for class_dir in subset_path.iterdir():
                if class_dir.is_dir():
                    class_name = class_dir.name
                    if class_name not in all_images:
                        all_images[class_name] = []
                    files = [f for f in class_dir.glob("*") if f.is_file()]
                    all_images[class_name].extend(files)
    
    if not all_images:
        logging.warning("No images found in %s", in_dir)
        return
    
    # Create output directories for each split and class
    for split in ["train", "val", "test"]:
        for class_name in all_images.keys():
            (out_dir / split / class_name).mkdir(parents=True, exist_ok=True)
    
    # Split each class independently (stratified at class level)
    manifest = {"ratios": ratios, "seed": seed, "counts": {}}
    for class_name, files in all_images.items():
        random.shuffle(files)
        n = len(files)
        n_train = int(ratios[0] * n)
        n_val = int(ratios[1] * n)
        
        splits = {
            "train": files[:n_train],
            "val": files[n_train:n_train + n_val],
            "test": files[n_train + n_val:]
        }
        
        for split, flist in splits.items():
            for f in flist:
                dst = out_dir / split / class_name / f.name
                if mode == "copy":
                    shutil.copy2(f, dst)
                else:
                    shutil.copy(f, dst)
        
        manifest["counts"][class_name] = {k: len(v) for k, v in splits.items()}
    
    (out_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2))
    logging.info("Split done: %s", manifest["counts"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--train", type=float, default=TRAIN_RATIO)
    parser.add_argument("--val", type=float, default=VAL_RATIO)
    parser.add_argument("--test", type=float, default=TEST_RATIO)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--mode", type=str, default="copy")
    args = parser.parse_args()
    setup_logger()
    split_dataset(args.in_dir, args.out_dir, (args.train, args.val, args.test), args.seed, args.mode)


if __name__ == "__main__":
    main()
