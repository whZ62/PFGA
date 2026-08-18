# Monthly data format

PFGA reads one PyTorch file per month:

```text
data/<dataset>/monthly/
  YYYY-MM/
    graph_data.pt
```

Each file should contain a plain dictionary of tensors. Worker/task indices in edge
tensors are local to that month; `worker_ids` are stable global IDs across months.

## Required fields in strict mode

| Field | Shape | Dtype | Meaning |
| --- | ---: | --- | --- |
| `worker_features` | `[m, d_w]` | float | location/time, preference, and history features |
| `task_features` | `[n, d_t]` | float | location/time and category features |
| `positive_edge_index` | `[2, E+]` | long | historical completed worker-task pairs |
| `social_edge_index` | `[2, E_s]` | long | directed representation of the undirected worker social graph |
| `benefit_matrix` | `[m, n]` | float | non-negative `u_ij` |
| `cost_matrix` | `[m, n]` | float | non-negative travel distance in kilometres |
| `worker_ids` | `[m]` | long | unique, stable, non-negative global worker IDs |
| `worker_capacity` | `[m]` | long | `q_i`; omitted values default to one |
| `preference_matrix` | `[m, n]` | float | `sim(p_i, c_j)` |
| `slack_matrix` | `[m, n]` | float | temporal slack `kappa_ij` from Equation (20) |
| `feasible_edge_index` | `[2, F_t]` | long | pairs satisfying Equation (17) |

`social_edge_index` can be empty. For an undirected relationship, include both
directions because attention messages follow the listed direction.

## Derivable alternatives

- `preference_matrix` can be replaced by `worker_preferences [m,p]` and
  `task_categories [n,p]`; cosine similarity is derived.
- `feasible_edge_index` can be replaced by `feasible_mask [m,n]`.
- Feasibility and slack can be derived from `worker_available_time [m]`,
  `task_time [n]`, and `worker_speed [m]`, provided time, distance, and speed use
  compatible units.
- `positive_edge_index` can be replaced by dense `labels` or `y` reshaped to `[m,n]`.

The loader also recognizes legacy names such as `worker_feat`, `task_feat`,
`worker_task_edge_index`, and `worker_worker_edge_index`. Set
`strict_paper_schema: false` only while migrating old data. In that mode, absent
`worker_ids` assume stable row order and absent feasibility treats all pairs as
reachable; both assumptions are warnings and should not be used for reported results.

## Minimal example

```python
torch.save(
    {
        "worker_features": worker_features.float(),
        "task_features": task_features.float(),
        "positive_edge_index": positive_edges.long(),
        "social_edge_index": social_edges.long(),
        "feasible_edge_index": feasible_edges.long(),
        "benefit_matrix": utility.float(),
        "cost_matrix": distance_km.float(),
        "preference_matrix": preference_similarity.float(),
        "slack_matrix": temporal_slack.float(),
        "worker_ids": global_worker_ids.long(),
        "worker_capacity": capacity.long(),
    },
    "data/yelp/monthly/2020-01/graph_data.pt",
)
```

Use `scripts/create_toy_data.py` as a complete executable example.

