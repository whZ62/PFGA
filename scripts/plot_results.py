#!/usr/bin/env python3
"""Plot cumulative outcomes from the exact same run used by summary.json."""

import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument("--run", required=True)
a = p.parse_args()
root = Path(a.run)
s = json.loads((root / "summary.json").read_text())
rows = json.loads((root / "test_trajectory.json").read_text())
if not rows:
    raise ValueError("No trajectory")
for key in ["utility", "fair", "assignment_rate", "cost"]:
    if not np.isclose(rows[-1][key], s["test"][key]):
        raise ValueError(f"Final metric mismatch: {key}")
released = np.array([x["released_tasks"] for x in rows])
indices = []
for target in np.linspace(1, released[-1], 100):
    candidates = np.flatnonzero(released <= target)
    indices.append(candidates[-1] if len(candidates) else 0)
fig, ax = plt.subplots(1, 2, figsize=(9, 3))
for a, key in zip(ax, ["utility", "fair"]):
    a.plot(np.arange(1, 101), [rows[i][key] for i in indices])
    a.set(xlabel="Replay progress (100 bins)", ylabel=key)
    a.grid(alpha=0.2)
fig.tight_layout()
fig.savefig(root / "temporal_metrics.png", dpi=200)
fig.savefig(root / "temporal_metrics.pdf")
