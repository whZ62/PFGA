# PFGA 2.0 — current WWW manuscript implementation

Aligned with *Dynamic Fairness Deficit Compensation for Personalized Task Assignment in
Spatial Crowdsourcing*, supplied revision `(6)`, 2026-10-25. This replaces the older
antibody-first / probability-weighted-utility implementation. It does not reproduce the
paper's numerical tables without the authors' processed datasets and full experiment settings.

## Quick start (Windows / Linux)

Use Python 3.9+ and install the PyTorch build appropriate for your CPU/CUDA environment,
then run from this directory:

```bash
python -m pip install -e ".[dev]"
python scripts/create_toy_data.py --config configs/toy.yaml
python -m pfga.cli --config configs/toy.yaml
python -m pytest -q
```

`toy` uses synthetic CSV interactions and is only an installation/integration check.
For a GPU, set `experiment.device: cuda` in YAML. Epoch logs and 100-decision progress
messages report training progress. Model selection uses validation AUC; the fixed
assignment/classification threshold is subsequently selected using validation ACC.
Test parameters remain frozen. All three replay splits start with fresh utility and
fairness memory; see the disclosed implementation choices below.

## Real data

Prepare a CSV with `worker_id,poi_id,category,timestamp,latitude,longitude` and optional
`retained` (1/0 annotation acceptance). Social CSV uses `worker_id,neighbor_id` and an
optional edge `timestamp`. See [docs/DATA_FORMAT.md](docs/DATA_FORMAT.md).

```bash
python -m pfga.prepare --config configs/paper_foursquare.yaml --interactions data/foursquare.csv --social data/social.csv
python -m pfga.cli --config configs/paper_foursquare.yaml
```

For headerless three-file check-in input:

```bash
python scripts/convert_checkins.py --checkins data/checkins.txt --categories data/poi_categories.txt --coordinates data/poi_coos.txt --output data/foursquare.csv
```

Use `--unix-seconds` for Unix timestamps. Without it, use ISO timestamps with no embedded
whitespace in whitespace-separated input. Categories must have one row per POI; choose
one category explicitly for multi-category POIs. No trajectory features are used.
For Yelp, export individual review user/business/date records to the CSV schema;
aggregate business-level check-in counts cannot supply user/task labels.

| Dataset | Train | Validation | Test |
|---|---|---|---|
| Foursquare | 2012-04 to 2013-04 | 2013-05 to 2013-09 | 2013-10 to 2014-01 |
| Yelp | 2016-01 to 2018-12 | 2019-01 to 2019-12 | 2020-01 to 2020-12 |

The preprocessor fits category counts, prior, availability centers, capacities and
initial locations on **training records only**. They are frozen for validation/test.
It does not manufacture expert annotations. Missing `retained` is recorded explicitly
as unfiltered observed interactions in `metadata.json`.

## Main changes

- Eq33 preference utility: `u_ij = task_reward[j] * affinity[i, category[j]]`.
- Eq4–13: worker encoder → social attention → `[deficit, memory]` antibody →
  dimension-wise gate → pair MLP on `[worker, task, worker*task]`.
- Eq8–9: normalized mean-utility deficit with signed bounded transform; EMA advances
  once per event decision. Fairness state receives **hard allocated utility**, not
  predicted probability mass. Every epoch and independent split resets it.
- Eq16–22: weighted BCE with weight-sum normalization; thresholded soft allocations
  respect task mass and worker capacity; fairness loss includes cumulative history.
- Chronological release, completion, expiry and availability-window events; dynamic
  worker location and busy time; no forced below-threshold assignments.
- Utility is preference-adjusted reward coverage; Fair uses cumulative allocated utility.
  Reward Gini and opportunity-normalized Gini are also produced.
- AUC/ACC use fixed reference feasible samples with a target 1:2 positive/negative ratio,
  independent of policy top-K. Unscored edges are never assigned fabricated zero scores.
- `cost` is total km; `time` is seconds per released task. Units and timing scope are
  recorded. Final table metrics and the temporal plots share the same accumulated data.

## Outputs and experiments

`outputs/...` contains resolved config, environment, history, best/last checkpoints,
validation threshold sweep, summary, cumulative test trajectory and per-worker outcomes.
Runtime fairness buffers in checkpoints are zeroed; learned parameters are preserved.
Previous 1.x checkpoints are architecture-incompatible and must be retrained.

```bash
python scripts/run_ablation.py --config configs/paper_yelp.yaml
python scripts/run_sensitivity.py --config configs/paper_yelp.yaml --parameter fairness_weight --values 0 0.25 0.5 1 2
python scripts/run_seeds.py --config configs/paper_yelp.yaml --seeds 1 2 3 4 5
python -m pip install -e ".[plots]"
python scripts/plot_results.py --run outputs/pfga_yelp
```

Ablations include the manuscript's components, Deficit-Score / Memory-Score / Joint-Score,
and an additional state-concatenation control. The score variants' exact formulas are
implementation choices because the PDF does not specify them; tune their coefficients
using validation. Seed aggregation reports sample standard deviation (`ddof=1`).
Re-preprocess after changing simulation parameters; changing a YAML simulation value
alone does not rewrite existing prepared streams.

## Compatibility and limits

The old monthly `graph_data.pt` path remains available with `data.mode: legacy_monthly`,
but requires explicit `task_rewards` to compute the new Utility metric. It cannot infer
true timestamps, locations or deadlines from matrices; its output is marked as a static
legacy protocol and must not be used to claim event-replay results. Use original records
and the new preprocessor for paper experiments.

See [docs/PAPER_ALIGNMENT.md](docs/PAPER_ALIGNMENT.md) for equation mappings and
unspecified-paper choices, and [AUDIT_REPORT_ZH.md](AUDIT_REPORT_ZH.md) for the migration
report. CPU tests and toy runs are verification, not Foursquare/Yelp benchmark replication.
The supplied package did not contain the ten comparison baselines or benchmark data;
this update implements PFGA and its variants, not unprovided baseline reproductions.
