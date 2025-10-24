#!/usr/bin/env python3
"""
mlops_xoxo/data_ingest.py

Dataset ingestion (Step 1 of the pipeline), aligned with your teammate's example.
- Simple style: constants at the top, `main()` drives ingestion.
- Supports downloading a ZIP from Hugging Face (raw URL `/resolve/main/...`) or using a local file.
- Extracts into `data/raw/<name>` and creates a manifest JSON for reproducibility.

Usage:
  python mlops_xoxo/data_ingest.py                  # use default HF URL and extract
  python mlops_xoxo/data_ingest.py --local val.zip  # use a local zip
  python mlops_xoxo/data_ingest.py --force          # overwrite output
  python mlops_xoxo/data_ingest.py --sha256 <HASH>  # optional integrity check

DVC integration (suggested):
  stages:
    ingest:
      cmd: python mlops_xoxo/data_ingest.py
      deps:
        - mlops_xoxo/data_ingest.py
      outs:
        - data/raw/deepfake_val

"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

# ======== Constants (adjust if needed) ========
DATA_URL = (
    "https://huggingface.co/datasets/pujanpaudel/deepfake_face_classification/resolve/main/val.zip"
)
CACHE_DIR = Path("data/external/hf_cache")
ARCHIVE_NAME = "val.zip"
OUTPUT_DIR = Path("data/raw/deepfake_val")  # follow data/raw convention
MANIFEST_NAME = "ingest_manifest.json"


# ======== Simple utils (no extra deps) ========
def setup_logger(verbosity: int = 1) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def sha256sum(path: Path, bufsize: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(bufsize)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        logging.info("File already cached: %s", dest)
        return dest
    logging.info("Downloading %s", url)
    with urllib.request.urlopen(url) as r, dest.open("wb") as f:
        shutil.copyfileobj(r, f)
    return dest


def extract_zip(archive_path: Path, out_dir: Path, overwrite: bool = False) -> Path:
    if out_dir.exists() and overwrite:
        logging.info("Removing previous output: %s", out_dir)
        shutil.rmtree(out_dir)
    if out_dir.exists() and not overwrite:
        logging.info("Output already exists, skipping extraction: %s", out_dir)
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.info("Extracting %s -> %s", archive_path, out_dir)
    with zipfile.ZipFile(archive_path, "r") as zf:
        zf.extractall(out_dir)
    return out_dir


def write_manifest(out_dir: Path, origin: str, archive_path: Path, expected_sha256: str | None) -> None:
    file_count = sum(1 for p in out_dir.rglob("*") if p.is_file())
    manifest = {
        "origin": origin,
        "archive_path": str(archive_path.resolve()),
        "output_dir": str(out_dir.resolve()),
        "file_count": file_count,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sha256": sha256sum(archive_path) if archive_path.exists() else None,
        "expected_sha256": expected_sha256,
        "python": sys.version,
        "argv": sys.argv,
    }
    (out_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logging.info("Manifest written to %s", out_dir / MANIFEST_NAME)


# ======== CLI ========
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ingest dataset: download and extract ZIP")
    p.add_argument("--url", type=str, default=DATA_URL, help="ZIP URL (HF /resolve/main/...)")
    p.add_argument("--out-dir", type=Path, default=OUTPUT_DIR, help="Output directory (extracted)")
    p.add_argument("--cache-dir", type=Path, default=CACHE_DIR, help="Cache directory for the ZIP")
    p.add_argument("--archive-name", type=str, default=ARCHIVE_NAME, help="Cache file name")
    p.add_argument("--local", type=Path, default=None, help="Use local ZIP instead of downloading")
    p.add_argument("--force", action="store_true", help="Overwrite output if exists")
    p.add_argument("--sha256", type=str, default=None, help="Expected SHA256 (optional)")
    p.add_argument("-v", "--verbose", action="count", default=1, help="-v / -vv for more logs")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    setup_logger(args.verbose)

    cache_dir: Path = args.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    if args.local is not None:
        archive_path = args.local
        origin = str(args.local)
    else:
        archive_path = cache_dir / args.archive_name
        origin = args.url
        download(args.url, archive_path)

    if args.sha256:
        actual = sha256sum(archive_path)
        if actual.lower() != args.sha256.lower():
            logging.error("SHA256 mismatch: expected=%s actual=%s", args.sha256, actual)
            return 3
        logging.info("SHA256 verified: %s", actual)

    out_dir: Path = args.out_dir
    extract_zip(archive_path, out_dir, overwrite=args.force)
    write_manifest(out_dir, origin, archive_path, args.sha256)

    logging.info("Ingestion complete. Files available at: %s", out_dir.resolve())
    print(str(out_dir.resolve()))  # for downstream scripts
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
