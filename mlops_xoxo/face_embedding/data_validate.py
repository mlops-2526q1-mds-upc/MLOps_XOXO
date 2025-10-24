import os
import json
from pathlib import Path
import cv2

RAW_DIR = "data/raw/face_embedding"           # per-person folders
MANIFEST_PATH = "data/processed/face_embedding/splits/manifest.json"
DOC_DIR = "docs/face_embedding"           # folder to save reports

os.makedirs(DOC_DIR, exist_ok=True)
report_path = os.path.join(DOC_DIR, "data_validation_report.txt")


def check_person_folders(raw_dir):
    """Function that checks people image information (and warning if needed)"""
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
    """Set if image is valid"""
    raw_dir = Path(raw_dir)
    corrupted = []
    for p in raw_dir.iterdir():
        if not p.is_dir():
            continue
        for img_path in p.glob("*.jpg"):
            img = cv2.imread(str(img_path))  # pylint: disable=no-member
            if img is None or img.size == 0:
                corrupted.append(str(img_path))
    return corrupted


def check_manifest(manifest_path):
    """Function that checks if there is a manifest and returns it"""
    if not os.path.exists(manifest_path):
        return ["Manifest not found"]

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    lines = []
    for split in ["train", "val", "test"]:
        images = manifest.get(split, [])
        lines.append(f"{split}: {len(images)} images")
    return lines


def data_validate_main():
    """Orchestrates data validation."""
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
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"Data validation report saved to {report_path}")


if __name__ == "__main__":
    data_validate_main()
