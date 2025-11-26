#!/usr/bin/env python3
"""
Preprocess dataset: face detection, cropping, resizing, normalization.

Input:
  --in-dir   Path to dataset with class subfolders (e.g., raw/val/{real,fake}/).
Output:
  --out-dir  Path to processed dataset with same class structure.
"""

import argparse, json, logging, time, sys, dataclasses
from pathlib import Path
import cv2, numpy as np, pandas as pd
from PIL import Image
import yaml

INPUT_DIR = Path("data/raw/fake_classification/val")
OUTPUT_DIR = Path("data/processed/fake_classification/val")

with open("pipelines/fake_classification/params.yaml", encoding="utf-8") as f:
    params = yaml.safe_load(f)

MARGIN = params["preprocessing"].get("margin", 0.25)
IMAGE_SIZE = params["preprocessing"].get("image_size", 224)

def setup_logger():
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")

def get_cascade():
    return cv2.CascadeClassifier(str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"))

def detect_face(img, cascade, min_size=30):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(min_size, min_size))
    if len(faces) == 0:
        return None
    return max(faces, key=lambda b: b[2] * b[3])

def crop_resize(img, bbox, size):
    x, y, w, h = bbox
    crop = img[y:y+h, x:x+w]
    return cv2.resize(crop, (size, size))

def save_image(arr, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)

def preprocess(in_dir: Path, out_dir: Path, size=224, margin=0.25):
    cascade = get_cascade()
    records = []
    skipped = []
    sums, sqsums, count = np.zeros(3), np.zeros(3), 0

    for cls_dir in in_dir.iterdir():
        if not cls_dir.is_dir(): continue
        for img_path in cls_dir.glob("*"):
            img = cv2.imread(str(img_path))
            if img is None:
                skipped.append(str(img_path)); continue
            bbox = detect_face(img, cascade)
            if bbox is None:
                skipped.append(str(img_path)); continue
            x,y,w,h = bbox
            cx,cy = x+w/2,y+h/2
            side = max(w,h)*(1+margin*2)
            x0,y0 = max(0,int(cx-side/2)), max(0,int(cy-side/2))
            x1,y1 = min(img.shape[1],int(cx+side/2)), min(img.shape[0],int(cy+side/2))
            crop = img[y0:y1, x0:x1]
            proc = cv2.resize(crop,(size,size))
            rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)
            save_path = out_dir/cls_dir.name/img_path.name
            save_image(rgb, save_path)
            # stats
            arr = rgb.astype(np.float32)/255.
            sums += arr.mean((0,1))
            sqsums += (arr**2).mean((0,1))
            count+=1
            records.append((str(img_path),str(save_path)))

    stats = {"mean": (sums/count).tolist(), "std": (np.sqrt(sqsums/count - (sums/count)**2)).tolist()}
    (out_dir/"normalization.json").write_text(json.dumps(stats,indent=2))
    pd.DataFrame(records,columns=["src","dst"]).to_csv(out_dir/"index.csv",index=False)
    (out_dir/"skipped.jsonl").write_text("\n".join(skipped))
    logging.info("Processed %d, skipped %d", len(records), len(skipped))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--margin", type=float, default=MARGIN)
    args = parser.parse_args()
    setup_logger()
    preprocess(args.in_dir, args.out_dir, args.size, args.margin)

if __name__ == "__main__":
    main()
