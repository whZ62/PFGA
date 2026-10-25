"""Build train-only preference profiles and immutable release streams from normalized CSV."""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from pfga.config import load_config


def prepare(config, interactions, social=None):
    sim = config.simulation
    df = pd.read_csv(interactions, dtype={"worker_id": str, "poi_id": str, "category": str})
    required = {"worker_id", "poi_id", "category", "timestamp", "latitude", "longitude"}
    if not required <= set(df):
        raise ValueError(f"Missing CSV columns: {required - set(df)}")
    if df[list(required)].isna().any().any():
        raise ValueError("Missing identity, category, coordinate or timestamp")
    if "retained" in df:
        df = df[df.retained.astype(str).str.lower().isin(["1", "true"])].copy()
    df["timestamp"] = pd.to_datetime(df.timestamp, utc=True, errors="raise")
    df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
    df["period"] = df.timestamp.dt.strftime("%Y-%m")
    df = df[(df.period >= config.data.train.start) & (df.period <= config.data.test.end)].copy()
    train = df[(df.period >= config.data.train.start) & (df.period <= config.data.train.end)]
    if train.empty:
        raise ValueError("No training records")
    if (
        not np.isfinite(df[["latitude", "longitude"]].to_numpy()).all()
        or (df.latitude.abs() > 90).any()
        or (df.longitude.abs() > 180).any()
    ):
        raise ValueError("Invalid coordinates")
    # ID inventory is metadata only. All behavioral statistics are fitted on train.
    workers = sorted(df.worker_id.unique())
    categories = sorted(train.category.unique())
    wmap, cmap = {x: i for i, x in enumerate(workers)}, {x: i for i, x in enumerate(categories)}
    unknown = len(categories)
    m, c = len(workers), len(categories) + 1
    counts = np.zeros((m, c), dtype=np.float32)
    for w, k in zip(train.worker_id, train.category):
        counts[wmap[w], cmap[k]] += 1
    prior = counts.sum(0) / counts.sum()
    affinity = (counts + prior) / (counts + prior).max(1, keepdims=True).clip(1e-8)
    coords = np.tile(train[["latitude", "longitude"]].mean().to_numpy(), (m, 1))
    center = np.full(m, 12.0, dtype=np.float32)
    capacity = np.ones(m, dtype=np.int64)
    days = max(
        1,
        (
            pd.Period(config.data.train.end).end_time
            - pd.Period(config.data.train.start).start_time
        ).days
        + 1,
    )
    for w, g in train.groupby("worker_id"):
        i = wmap[w]
        coords[i] = g[["latitude", "longitude"]].iloc[-1]
        hour = (
            g.timestamp.dt.hour + g.timestamp.dt.minute / 60 + g.timestamp.dt.second / 3600
        ).to_numpy()
        center[i] = (
            np.arctan2(np.sin(hour * np.pi / 12).mean(), np.cos(hour * np.pi / 12).mean())
            * 12
            / np.pi
            % 24
        )
        in_window = np.abs((hour - center[i] + 12) % 24 - 12) < sim.availability_hours / 2
        capacity[i] = min(sim.capacity_cap, max(1, int(np.ceil(in_window.sum() / days))))
    edges = []
    if social:
        sf = pd.read_csv(social, dtype=str)
        if "timestamp" in sf:
            cutoff = pd.Period(config.data.train.end).end_time.tz_localize("UTC")
            sf = sf[pd.to_datetime(sf.timestamp, utc=True) <= cutoff]
        for a, b in zip(sf.worker_id, sf.neighbor_id):
            if a in wmap and b in wmap and a != b:
                edges.extend([(wmap[a], wmap[b]), (wmap[b], wmap[a])])
    edges = np.array(sorted(set(edges)), dtype=np.int64).reshape(-1, 2).T
    # Retrieval uses a simple documented social-enriched HISTORICAL profile; utility never does.
    social_profile = counts / (counts.sum(1, keepdims=True) + 1e-8)
    agg = np.zeros_like(social_profile)
    degree = np.zeros(m)
    for a, b in edges.T:
        agg[b] += social_profile[a]
        degree[b] += 1
    profile = social_profile + agg / degree.clip(1)[:, None]
    pref = profile / (np.linalg.norm(profile, axis=1, keepdims=True) + 1e-8)
    wf = np.concatenate(
        [
            counts / (counts.sum(1, keepdims=True) + 1e-8),
            coords / np.array([90, 180]),
            np.sin(center[:, None] * np.pi / 12),
            np.cos(center[:, None] * np.pi / 12),
            np.log1p(counts.sum(1))[:, None],
        ],
        1,
    )
    common = {
        "worker_features": torch.tensor(wf, dtype=torch.float32),
        "initial_locations": torch.tensor(coords, dtype=torch.float64),
        "centers": torch.tensor(center, dtype=torch.float64),
        "capacities": torch.tensor(capacity),
        "affinity": torch.tensor(affinity),
        "retrieval_profile": torch.tensor(pref, dtype=torch.float32),
        "social_edge_index": torch.tensor(edges),
        "num_categories": c,
        "speed_kmh": sim.speed_kmh,
        "window_hours": sim.availability_hours,
        "deadline_hours": sim.deadline_hours,
    }
    root = Path(config.data.directory)
    root.mkdir(parents=True, exist_ok=True)
    gen = np.random.default_rng(config.data.sampling_seed)
    df["reward"] = gen.uniform(sim.reward_min, sim.reward_max, len(df))
    for name in ["train", "validation", "test"]:
        split = getattr(config.data, name)
        part = df[(df.period >= split.start) & (df.period <= split.end)]
        if part.empty:
            raise ValueError(f"Empty {name} split")
        stream = []
        for t, row in enumerate(part.itertuples()):
            stream.append(
                {
                    "id": t,
                    "time": row.timestamp.timestamp() / 3600,
                    "worker": wmap[row.worker_id],
                    "category": cmap.get(row.category, unknown),
                    "location": [row.latitude, row.longitude],
                    "reward": row.reward,
                }
            )
        torch.save({"common": common, "tasks": stream, "split": name}, root / f"{name}.pt")
    metadata = {
        "schema_version": 2,
        "workers": m,
        "categories": categories + ["<unknown>"],
        "worker_ids": workers,
        "train_rows": len(train),
        "split_counts": {
            n: int(
                (
                    (df.period >= getattr(config.data, n).start)
                    & (df.period <= getattr(config.data, n).end)
                ).sum()
            )
            for n in ["train", "validation", "test"]
        },
        "statistics_source": "training only; frozen in all replay splits",
        "annotation_filter": "retained column"
        if "retained" in df
        else "no annotation supplied; all observed records retained",
        "time_basis": "UTC; convert local timestamps to explicit UTC offsets before import",
        "new_worker_profile": "prior affinity; mean train location; noon; capacity one",
        "reference_locations": "fixed train last location; common across methods",
        "sampling_seed": config.data.sampling_seed,
        "simulation": vars(sim),
    }
    (root / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--interactions", required=True)
    parser.add_argument("--social")
    a = parser.parse_args()
    print(json.dumps(prepare(load_config(a.config), a.interactions, a.social), indent=2))


if __name__ == "__main__":
    main()
