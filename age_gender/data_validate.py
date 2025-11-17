"""
data_validate.py - Validate dataset quality
"""

import pandas as pd
from pathlib import Path
from PIL import Image
import json


# Configuration
DATA_DIR = Path("data/raw/utkface_aligned_cropped/UTKFace")  # ← Corrigé
METADATA_FILE = Path("data/processed/utkface_metadata.csv")
OUTPUT_DIR = Path("data/processed/validation")


def main():
    print("=" * 60)
    print("DATA VALIDATION")
    print("=" * 60)
    
    # Load metadata
    df = pd.read_csv(METADATA_FILE)
    print(f"\nChecking {len(df)} images...")
    
    # Check images
    corrupted = []
    missing = []
    
    for idx, row in df.iterrows():
        img_path = DATA_DIR / row['filename']
        
        if not img_path.exists():
            missing.append(row['filename'])
            continue
        
        try:
            img = Image.open(img_path)
            img.verify()
        except:
            corrupted.append(row['filename'])
    
    # Results
    valid = len(df) - len(corrupted) - len(missing)
    print(f"\nResults:")
    print(f"  Valid:     {valid}")
    print(f"  Corrupted: {len(corrupted)}")
    print(f"  Missing:   {len(missing)}")
    
    # Stats
    print(f"\nDistribution:")
    print(f"  Age: {df['age'].min()}-{df['age'].max()} (mean: {df['age'].mean():.1f})")
    print(f"  Gender: {df['gender'].value_counts().to_dict()}")
    
    # Save report
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        'total': len(df),
        'valid': valid,
        'corrupted': len(corrupted),
        'missing': len(missing),
        'status': 'PASSED' if len(corrupted) == 0 else 'WARNING'
    }
    
    report_path = OUTPUT_DIR / "validation_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
    
    print(f"\nReport saved to: {report_path}")
    print(f"Status: {report['status']}")
    
    return 0 if report['status'] == 'PASSED' else 1


if __name__ == "__main__":
    exit(main())