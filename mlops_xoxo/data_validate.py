import os
from pathlib import Path
import cv2
import json

RAW_DIR = "data/raw"           # per-person folders
MANIFEST_PATH = "data/processed/splits/manifest.json"
DOC_DIR = "docs"           # folder to save reports

os.makedirs(DOC_DIR, exist_ok=True)
report_path = os.path.join(DOC_DIR, "data_validation_report.txt")

def check_person_folders(raw_dir):
    raw_dir = Path(raw_dir)
    person_folders = [p for p in raw_dir.iterdir() if p.is_dir()]

    total_images = 0
    warnings = []
    for p in person_folders:
        images = list(p.glob("*.jpg"))
        if len(images) == 0:
            warnings.append(f"Warning: Person {p.name} has no images")
        total_images += len(images)

    avg_per_person = total_images / len(person_folders) if person_folders else 0
    summary = [
        f"Found {len(person_folders)} persons",
        f"Total images: {total_images}",
        f"Average images per person: {avg_per_person:.2f}"
    ]
    return summary + warnings

def validate_images(raw_dir):
    raw_dir = Path(raw_dir)
    corrupted = []
    for p in raw_dir.iterdir():
        if not p.is_dir():
            continue
        for img_path in p.glob("*.jpg"):
            img = cv2.imread(str(img_path))
            if img is None:
                corrupted.append(str(img_path))
    return corrupted

def check_manifest(manifest_path):
    if not os.path.exists(manifest_path):
        return ["Manifest not found"]

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    lines = []
    for split in ["train", "val", "test"]:
        images = manifest.get(split, [])
        lines.append(f"{split}: {len(images)} images")
    return lines

if __name__ == "__main__":
    report_lines = []
    report_lines.append("=== Person folder check ===")
    report_lines += check_person_folders(RAW_DIR)
    report_lines.append("\n=== Image integrity check ===")
    corrupted = validate_images(RAW_DIR)
    if not corrupted:
        report_lines.append("All images are valid!")
    else:
        report_lines.append(f"Corrupted images: {len(corrupted)}")
        report_lines += corrupted
    report_lines.append("\n=== Manifest check ===")
    report_lines += check_manifest(MANIFEST_PATH)

    # Save report
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))

    print(f"Data validation report saved to {report_path}")