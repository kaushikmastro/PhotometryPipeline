#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import pandas as pd


def collect_parquet_files(project_root: Path) -> list[Path]:
    """Collect survey + hamo geometry parquet files."""
    base_dir = project_root / "data" / "04_geometry_tables"
    survey_dir = base_dir / "survey"
    hamo_dir = base_dir / "hamo"

    files: list[Path] = []
    for folder in (survey_dir, hamo_dir):
        if folder.exists():
            files.extend(sorted(folder.rglob("*.parquet")))

    return files


def format_progress(current: int, total: int, width: int = 36) -> str:
    """Build a compact ASCII progress bar."""
    if total <= 0:
        return "[" + ("-" * width) + "] 0/0"

    ratio = current / total
    filled = int(ratio * width)
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {current}/{total} ({ratio * 100:6.2f}%)"


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    parquet_files = collect_parquet_files(project_root)

    if not parquet_files:
        print("No parquet files found under data/04_geometry_tables/survey or hamo.")
        return 1

    print(f"Found {len(parquet_files)} parquet files to patch.")

    processed = 0
    skipped = 0

    for idx, parquet_path in enumerate(parquet_files, start=1):
        try:
            df = pd.read_parquet(parquet_path, engine="pyarrow")

            if "iof" not in df.columns:
                skipped += 1
                print(f"\nSKIP (missing iof): {parquet_path}")
                continue

            df["iof"] = df["iof"].astype("float32") / 100.0
            df.to_parquet(parquet_path, engine="pyarrow", index=False)
            processed += 1
        except Exception as exc:
            print(f"\nERROR: {parquet_path} -> {exc}")
            return 2
        finally:
            progress_line = format_progress(idx, len(parquet_files))
            print("\r" + progress_line, end="", flush=True)

    print()
    print("Surgical strike complete.")
    print(f"Successfully updated: {processed}")
    print(f"Skipped (missing iof): {skipped}")
    print("All targeted parquet files have been overwritten with corrected iof values.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
