#!/usr/bin/env python3
"""Convert headerless user/poi/time + POI category/coordinates into normalized CSV."""

import argparse
import pandas as pd
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--checkins", required=True, help="worker_id poi_id timestamp (3 columns)")
p.add_argument("--categories", required=True, help="poi_id category (2 columns)")
p.add_argument("--coordinates", required=True, help="poi_id latitude longitude (3 columns)")
p.add_argument("--output", required=True)
p.add_argument(
    "--separator", default=r"\s+", help="Default whitespace; use , for CSV or \\t for TSV"
)
p.add_argument("--unix-seconds", action="store_true")
a = p.parse_args()
x = pd.read_csv(
    a.checkins, sep=a.separator, header=None, names=["worker_id", "poi_id", "timestamp"], dtype=str
)
c = pd.read_csv(a.categories, sep=a.separator, header=None, names=["poi_id", "category"], dtype=str)
g = pd.read_csv(
    a.coordinates,
    sep=a.separator,
    header=None,
    names=["poi_id", "latitude", "longitude"],
    dtype=str,
)
if c.poi_id.duplicated().any():
    raise ValueError(
        "Choose one category per POI before conversion; duplicate categories are ambiguous"
    )
y = x.merge(c, on="poi_id", validate="many_to_one").merge(g, on="poi_id", validate="many_to_one")
if len(y) != len(x):
    raise ValueError("Missing POI category or coordinates; no silent row dropping")
if a.unix_seconds:
    y["timestamp"] = pd.to_datetime(y.timestamp.astype(float), unit="s", utc=True).astype(str)
Path(a.output).parent.mkdir(parents=True, exist_ok=True)
y.to_csv(a.output, index=False)
print(f"Converted {len(y)} interactions to {a.output}")
