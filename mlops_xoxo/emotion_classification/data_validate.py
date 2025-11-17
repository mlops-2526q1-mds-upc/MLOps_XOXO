#!/usr/bin/env python3
"""Basic data validation for emotion dataset.

Generates a small report under `docs/emotion/data_validation_report.txt` describing class counts
and missing files. This mirrors other pipelines' lightweight validation steps.
"""
from pathlib import Path
import argparse
import textwrap
import yaml


def gather_stats(root: Path):
    stats = {}
    for c in sorted([p for p in root.iterdir() if p.is_dir()]):
        stats[c.name] = len(list(c.glob("*")))
    return stats


def main():
    with open("pipelines/emotion_classification/params.yaml", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    raw_dir = Path(params["dataset"]["raw_dir"]) / "emotion"
    out_dir = Path("docs/emotion")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw dir not found: {raw_dir}")

    stats = gather_stats(raw_dir)
    report = textwrap.dedent(f"""
    Emotion dataset validation
    ==========================
    raw_dir: {raw_dir}

    class counts:
    """)
    for k, v in stats.items():
        report += f"    {k}: {v}\n"

    (out_dir / "data_validation_report.txt").write_text(report)
    print(str(out_dir / "data_validation_report.txt"))


if __name__ == "__main__":
    main()
