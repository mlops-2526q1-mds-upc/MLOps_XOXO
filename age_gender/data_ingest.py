"""
data_ingest.py - Parse UTKFace filenames and create metadata CSV
"""

import pandas as pd
from pathlib import Path
import json


# Configuration
DATA_DIR = Path("data/raw/utkface_aligned_cropped/UTKFace") 
OUTPUT_DIR = Path("data/processed")
METADATA_FILE = "utkface_metadata.csv"


def parse_filename(filename):
    """Parse UTKFace filename: [age]_[gender]_[race]_[timestamp].jpg"""
    try:
        parts = filename.split('_')
        return {
            'filename': filename,
            'age': int(parts[0]),
            'gender': int(parts[1]),  # 0=Female, 1=Male
            'race': int(parts[2])     # 0-4
        }
    except:
        return None


def main():
    print("=" * 60)
    print("DATA INGESTION")
    print("=" * 60)
    
    # List all images
    image_files = list(DATA_DIR.glob("*.jpg"))
    print(f"\nFound {len(image_files)} images in {DATA_DIR}")
    
    # Parse filenames
    print("Parsing metadata...")
    data = []
    errors = 0
    
    for img_file in image_files:
        parsed = parse_filename(img_file.name)
        if parsed:
            data.append(parsed)
        else:
            errors += 1
    
    df = pd.DataFrame(data)
    
    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / METADATA_FILE
    df.to_csv(output_path, index=False)
    
    print(f"Successfully parsed: {len(df)} images")
    print(f"Errors: {errors}")
    print(f"Saved to: {output_path}")
    
    # Stats
    print(f"\nAge range: {df['age'].min()}-{df['age'].max()}")
    print(f"Gender distribution: {df['gender'].value_counts().to_dict()}")
    
    return 0


if __name__ == "__main__":
    exit(main())