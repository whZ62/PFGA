# PFGA

Official-style release implementation of **Personalized Fair Graph Assigner (PFGA)**
for personalized and fairness-aware task assignment in spatial crowdsourcing.

The package follows Algorithm 1 and Equations (1)-(23) of the accompanying paper:

> *Antibody-Enhanced Fairness Adaptation for Personalized Task Assignment in
> Spatial Crowdsourcing*.


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


Run the paper's repeated seeds from the same validated base configuration:

```bash
python scripts/run_seeds.py --config configs/paper_yelp.yaml --seeds 1 2 3 4 5
```




