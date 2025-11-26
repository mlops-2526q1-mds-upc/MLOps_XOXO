"""
data_split.py - Split dataset into train/val/test
"""

import pandas as pd
from pathlib import Path
import yaml

with open("pipelines/age_gender/params.yaml", "r") as f:
    params = yaml.safe_load(f)

# Configuration
INPUT_FILE = Path(params['dataset']['processed_dir']) / "utkface_metadata.csv"
OUTPUT_DIR = Path(params['dataset']['processed_dir']) / "splits"
TRAIN_RATIO = params['dataset']['split']['train']
VAL_RATIO = params['dataset']['split']['val']
TEST_RATIO = params['dataset']['split']['test']
SEED = params['seed']


def main():
    print("=" * 60)
    print("DATA SPLIT")
    print("=" * 60)
    
    # Load data
    df = pd.read_csv(INPUT_FILE)
    print(f"\nLoaded {len(df)} images")
    
    # Shuffle with fixed seed
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    
    # Calculate split indices
    n = len(df)
    train_end = int(n * TRAIN_RATIO)
    val_end = train_end + int(n * VAL_RATIO)
    
    # Split
    train_df = df[:train_end]
    val_df = df[train_end:val_end]
    test_df = df[val_end:]
    
    print(f"\nSplit (seed={SEED}):")
    print(f"  Train: {len(train_df)} ({len(train_df)/n*100:.1f}%)")
    print(f"  Val:   {len(val_df)} ({len(val_df)/n*100:.1f}%)")
    print(f"  Test:  {len(test_df)} ({len(test_df)/n*100:.1f}%)")
    
    # Save splits
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(OUTPUT_DIR / "train.csv", index=False)
    val_df.to_csv(OUTPUT_DIR / "val.csv", index=False)
    test_df.to_csv(OUTPUT_DIR / "test.csv", index=False)
    
    print(f"\nSaved to: {OUTPUT_DIR}")
    
    return 0


if __name__ == "__main__":
    exit(main())