#!/usr/bin/env python3
"""
Split dataset into train/val/test.

Input:
  --in-dir   Path to processed dataset with class subfolders.
Output:
  --out-dir  Path to split dataset with train/val/test subfolders.
"""

import argparse, json, logging, shutil, random, time, sys
from pathlib import Path

import yaml


with open("pipelines/fake_classification/params.yaml", encoding="utf-8") as f:
    params = yaml.safe_load(f)
input_dir = Path(params["dataset"]["processed_dir"]) / "fake_classification/val"
output_dir = Path(params["dataset"]["processed_dir"]) / "fake_classification/split"
train_ratio = params["dataset"]["split"]["train"]
val_ratio = params["dataset"]["split"]["val"]
test_ratio = params["dataset"]["split"]["test"]
seed = params["seed"]

def setup_logger():
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")

def split_dataset(in_dir: Path, out_dir: Path, ratios, seed=42, mode="copy"):
    random.seed(seed)
    classes = [d for d in in_dir.iterdir() if d.is_dir()]
    manifest = {"ratios":ratios,"seed":seed,"counts":{}}

    for split in ["train","val","test"]:
        for c in classes:
            (out_dir/split/c.name).mkdir(parents=True, exist_ok=True)

    for c in classes:
        files = list(c.glob("*"))
        random.shuffle(files)
        n = len(files)
        n_train, n_val = int(ratios[0]*n), int(ratios[1]*n)
        splits = {"train":files[:n_train], "val":files[n_train:n_train+n_val], "test":files[n_train+n_val:]}
        for split,flist in splits.items():
            for f in flist:
                dst = out_dir/split/c.name/f.name
                if mode=="copy": shutil.copy2(f,dst)
                else: shutil.copy(f,dst)
        manifest["counts"][c.name] = {k:len(v) for k,v in splits.items()}

    (out_dir/"split_manifest.json").write_text(json.dumps(manifest,indent=2))
    logging.info("Split done: %s", manifest["counts"])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", type=Path, default=input_dir)
    parser.add_argument("--out-dir", type=Path, default=output_dir)
    parser.add_argument("--train", type=float, default=train_ratio)
    parser.add_argument("--val", type=float, default=val_ratio)
    parser.add_argument("--test", type=float, default=test_ratio)
    parser.add_argument("--seed", type=int, default=seed)
    parser.add_argument("--mode", type=str, default="copy")
    args = parser.parse_args()
    setup_logger()
    split_dataset(args.in_dir, args.out_dir, (args.train,args.val,args.test), args.seed, args.mode)

if __name__ == "__main__":
    main()
