# PFGA

Official-style release implementation of **Personalized Fair Graph Assigner (PFGA)**
for personalized and fairness-aware task assignment in spatial crowdsourcing.

The package follows Algorithm 1 and Equations (1)-(23) of the accompanying paper:

> *Antibody-Enhanced Fairness Adaptation for Personalized Task Assignment in
> Spatial Crowdsourcing*.

## What is included

- dynamic fairness-antibody memory with exponential smoothing (Eq. 1-3);
- gated worker encoder and independent task encoder (Eq. 4-7);
- residual multi-head social attention (Eq. 8-9);
- sparse feasible candidate graph materialization (Eq. 17-23);
- edge-only assignment prediction and the composite loss (Eq. 10, 12-16);
- adaptive-threshold, capacity-aware constrained greedy assignment (Eq. 11);
- chronological monthly training, validation, testing, checkpoints, and metrics;
- configurations for the exact paper time splits on Yelp and Foursquare;
- a deterministic toy-data path for installation checks.

The precise implementation decisions needed to make several ambiguous paper formulas
executable are disclosed in [docs/PAPER_ALIGNMENT.md](docs/PAPER_ALIGNMENT.md).
The supplied prototype audit is also summarized in Chinese in
[AUDIT_REPORT_ZH.md](AUDIT_REPORT_ZH.md).

## Installation

Python 3.9 or newer is required. Install PyTorch for your CUDA/CPU platform first,
then install the package:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

The implementation uses native PyTorch scatter operations and does not require
PyTorch Geometric.

## Smoke test

Create three synthetic monthly batches and run a short experiment:

```bash
python scripts/create_toy_data.py --output data/toy
pfga-train --config configs/toy.yaml
python -m pytest
```

Expected artifacts are written to `outputs/toy/`, including `best.pt`, `last.pt`,
`history.jsonl`, and `summary.json`.

## Paper experiments

Prepare monthly files using the schema in [docs/DATA_FORMAT.md](docs/DATA_FORMAT.md),
then set `data.directory` in the corresponding configuration:

```bash
pfga-train --config configs/paper_yelp.yaml
pfga-train --config configs/paper_foursquare.yaml
```

The configurations encode the paper's chronological splits:

| Dataset | Training | Validation | Test |
| --- | --- | --- | --- |
| Foursquare | 2012-04 to 2013-04 | 2013-05 to 2013-09 | 2013-10 to 2014-01 |
| Yelp | 2016-01 to 2018-12 | 2019-01 to 2019-12 | 2020-01 to 2020-12 |

The released paper does not specify the candidate budget or the four materialization
weights. They are therefore explicit configuration parameters, not hidden constants.
Run sensitivity analysis before claiming exact table reproduction.

Run the paper's ablations, one-dimensional sensitivity experiments, and repeated
seeds from the same validated base configuration:

```bash
python scripts/run_ablation.py --config configs/paper_yelp.yaml
python scripts/run_sensitivity.py --config configs/paper_yelp.yaml \
  --parameter lambda_p --values 0.1 0.3 0.5 0.7 0.9
python scripts/run_seeds.py --config configs/paper_yelp.yaml --seeds 1 2 3 4 5
```

Every variant gets an isolated output directory and an aggregate JSON summary.

## Reproducibility

- all random generators are seeded;
- deterministic PyTorch algorithms can be enabled from YAML;
- every run records the resolved configuration and environment metadata;
- model selection uses validation AUC; test data are evaluated once from `best.pt`;
- validation memory is isolated from training memory;
- candidate recall is reported to expose information lost during sparsification.

## Public-release note

The authors should choose and add a software license before publishing this repository.
No license is inferred here because licensing is an author/legal decision.

## Citation

Citation metadata are provided in [CITATION.cff](CITATION.cff). Add the final venue,
pages, and DOI once assigned.
