Preprocessed data: https://drive.google.com/file/d/1_3z8tMj6GCT0Lr1zj-Cpf_SB6xC5_K6u/view?usp=sharing
# Data format and migration

## Normalized CSV input

One row is one observed interaction and one released task instance. Repeated POIs are
separate tasks with unique instance IDs, not a single persistent task.

```csv
worker_id,poi_id,category,timestamp,latitude,longitude,retained
w1,p1,food,2020-01-01T12:00:00+00:00,-37.81,144.96,1
```

- IDs are strings; coordinates are degrees; distances are Haversine km.
- Use explicit timestamp offsets; UTC is the replay clock. Do not silently mix local
  timestamps from different regions. For local-time availability experiments, preprocess
  separate regions under an explicitly documented time convention.
- Categories are one ID per task. Multi-category inputs require an explicit upstream
  category mapping; the converter rejects duplicate POI category rows.
- Optional `retained` must be 1/0 or true/false. This imports author-provided annotations;
  the program does not assert that three experts annotated unfiltered data.
- Optional social CSV: `worker_id,neighbor_id,timestamp`; edges become undirected and
  deduplicated. Omit the file if no social graph exists. Future-dated edges are filtered.

## Prepared stream

`pfga.prepare` writes `train.pt`, `validation.pt`, `test.pt`, `metadata.json` under
`data.directory`. Each stream contains:

- `common`: train-only worker features, initial locations, daily centers, capacities,
  Eq33 affinity, social-enriched retrieval profile, social edges and simulation constants;
- `tasks`: sorted dictionaries with unique instance ID, release time (UTC hours), category,
  location, fixed reward, and observed worker (used only for labels, never candidate ranking);
- `split`: name.

Workers include the registered ID inventory; unseen-worker assumptions are disclosed in
PAPER_ALIGNMENT.md. Category vocabulary is train-only plus an unknown slot. The metadata
records split sizes and whether annotations were provided. Simulation parameters are
stored at preprocessing time. Rebuild streams after changing simulation settings.

## Legacy monthly compatibility

Set `data.mode: legacy_monthly`. The folder layout remains
`YYYY-MM/graph_data.pt`, and feature / edge names from 1.x are accepted. In addition to
`worker_features`, `task_features`, `worker_ids`, positive/social/feasible edges,
`benefit_matrix`, `cost_matrix`, `preference_matrix`, `slack_matrix`, `worker_capacity`,
you MUST provide `task_rewards` of shape [N].

`benefit_matrix` must already equal Eq33 `u_ij`; `preference_matrix` is retrieval
similarity, not the utility affinity. Do not substitute its cosine scores for Eq33.
`worker_ids` must be stable across months. All matrices/indices use local batch rows.
Without original category counts and raw task rewards, old benefit matrices cannot be
converted exactly; this code will not guess rewards by taking row/column maxima.

This adapter provides static monthly constrained assignments only. It cannot reconstruct
within-month mobility, completion events or deadlines; outputs mark this limitation.
Use raw interaction records and the stream preprocessor for the current paper protocol.
