# data_split.py
import os
import shutil
import json
from pathlib import Path
from mlops_xoxo.utils.data_utils import create_manifest

# Paths
RAW_DIR = Path("data/raw")              # raw images by person
MANIFEST_PATH = Path("data/processed/splits/manifest.json")  # manifest with splits
OUTPUT_DIR = Path("data/processed")              # root output folder

SPLITS = ["train", "val", "test"]


def split_data():
    """
    Orchestrates the data splitting and manifest creation process
    """
    # 1. Create a temporary "plan" based on the raw data.
    print("Step 1: Creating a temporary splitting plan from raw data...")
    temp_manifest_path = Path("temp_manifest.json")
    create_manifest(RAW_DIR, temp_manifest_path)
    with open(temp_manifest_path, encoding="utf-8") as f:
        raw_path_manifest = json.load(f)
    os.remove(temp_manifest_path)  # Clean up the temporary file

    # 2. Copy files to their new locations and build the final, correct manifest.
    print("\nStep 2: Copying files and building the final manifest...")
    final_manifest = {'train': [], 'val': [], 'test': []}

    for split in SPLITS:
        split_dir = OUTPUT_DIR / split

        for img_path_str in raw_path_manifest.get(split, []):
            img_path = Path(img_path_str)
            person_id = img_path.parent.name

            target_person_dir = split_dir / person_id
            target_person_dir.mkdir(parents=True, exist_ok=True)

            destination_path = target_person_dir / img_path.name
            shutil.copy(img_path, destination_path)

            final_manifest[split].append(str(destination_path))

        print(f"  - Copied {len(raw_path_manifest.get(split, []))} images to '{split}' split.")

    # 3. Save the final manifest with the correct 'data/processed' paths.
    print("\nStep 3: Saving the final manifest...")
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, 'w', encoding="utf-8") as f:
        json.dump(final_manifest, f, indent=2)

    print("\n Data split and manifest creation completed!")
    print(f"Correct manifest saved to: {MANIFEST_PATH}")


if __name__ == "__main__":
    split_data()
