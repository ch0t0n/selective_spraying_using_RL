#!/usr/bin/env python3
"""
merge_eval_results.py — Merge per-job evaluation CSVs into analysis-ready files.

Generated for the Beocat/jameschapman version of the repository.
It reads logs/results/tmp/... and writes:
  logs/results/ablation_reward.csv
  logs/results/ablation_uncertainty.csv
  logs/results/dr_inDist.csv
  logs/results/dr_OOD.csv
"""

import argparse
import os
from pathlib import Path
import pandas as pd


def merge_pattern(root: Path, pattern: str, out_path: Path) -> None:
    files = sorted(root.rglob(pattern)) if root.exists() else []
    if not files:
        print(f"[WARN] no files found for {root}/{pattern}")
        return

    frames = []
    for fp in files:
        try:
            frames.append(pd.read_csv(fp))
        except pd.errors.EmptyDataError:
            print(f"[WARN] empty CSV skipped: {fp}")
        except Exception as exc:
            print(f"[WARN] could not read {fp}: {exc}")

    if not frames:
        print(f"[WARN] no readable files for {root}/{pattern}")
        return

    df = pd.concat(frames, ignore_index=True).drop_duplicates()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path} with {len(df)} rows from {len(files)} files")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="logs/results")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    jobs = [
        (results_dir / "tmp" / "ablation_reward",      "result_*.csv", results_dir / "ablation_reward.csv"),
        (results_dir / "tmp" / "ablation_uncertainty", "result_*.csv", results_dir / "ablation_uncertainty.csv"),
        (results_dir / "tmp" / "dr",                   "inDist_*.csv", results_dir / "dr_inDist.csv"),
        (results_dir / "tmp" / "dr",                   "OOD_*.csv",    results_dir / "dr_OOD.csv"),
    ]

    for root, pattern, out_path in jobs:
        merge_pattern(root, pattern, out_path)


if __name__ == "__main__":
    main()
