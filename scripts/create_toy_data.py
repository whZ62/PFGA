#!/usr/bin/env python3
"""Synthetic input only; never substitutes for paper benchmark results."""

import argparse
import csv
from pathlib import Path
import numpy as np
from pfga.config import load_config
from pfga.prepare import prepare


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="data/toy_raw")
    p.add_argument("--config", default="configs/toy.yaml")
    a = p.parse_args()
    root = Path(a.output)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    with (root / "interactions.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["worker_id", "poi_id", "category", "timestamp", "latitude", "longitude"])
        for month in [1, 2, 3]:
            for day in range(1, 5):
                for i in range(6):
                    w.writerow(
                        [
                            f"w{i}",
                            f"p{i % 3}",
                            f"c{i % 3}",
                            f"2020-{month:02d}-{day:02d}T12:{i * 5:02d}:00Z",
                            -37.8 + rng.uniform(-0.003, 0.003),
                            144.95 + rng.uniform(-0.003, 0.003),
                        ]
                    )
    with (root / "social.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["worker_id", "neighbor_id"])
        w.writerows((f"w{i}", f"w{(i + 1) % 6}") for i in range(6))
    print(prepare(load_config(a.config), root / "interactions.csv", root / "social.csv"))


if __name__ == "__main__":
    main()
