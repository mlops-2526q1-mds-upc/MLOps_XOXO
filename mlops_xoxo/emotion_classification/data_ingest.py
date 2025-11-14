"""Simple ingestion for local emotion images.

This script assumes the raw images are already present under `data/raw/emotion`.
It writes a small manifest and ensures the directory exists so DVC stages can depend on it.
"""
from pathlib import Path
import json
import time


RAW_DIR = Path("data/raw/emotion")
MANIFEST = RAW_DIR / "ingest_manifest.json"


def write_manifest(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    file_count = sum(1 for p in out_dir.rglob("*") if p.is_file())
    manifest = {
        "output_dir": str(out_dir.resolve()),
        "file_count": file_count,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    print(str(MANIFEST))


def main():
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"Expected raw data at {RAW_DIR} (place your images under this path)")
    write_manifest(RAW_DIR)


if __name__ == "__main__":
    main()
