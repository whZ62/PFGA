# Alignment with manuscript revision (6), 2026-10-25

| Paper | Implementation | Correction from 1.x |
|---|---|---|
| Eq1 | losses.gini_coefficient / metrics.ReplayMetrics | cumulative allocated utility; includes zeros |
| Eq2–3 | replay.Replay / assignment.constrained_greedy_assignment | updated location, availability, deadline and per-window capacity |
| Eq4–7 | model.PFGA.encode / SocialPersonalizationAttention | social aggregation before fairness; separate attention and value projections |
| Eq8 | FairnessAntibodyMemory.deficit | `(mean-U)/(mean+eps)` then `x/(1+abs(x))`; no pairwise Frobenius normalization |
| Eq9 | FairnessAntibodyMemory.begin_step | EMA before scoring each decision; gamma=.9 |
| Eq10 | FairnessAntibodyGenerator | two input scalars and sigmoid-bounded adapter |
| Eq11–12 | PFGA.gate | vector gate after social personalization |
| Eq13 | PFGA.assignment_predictor | nonlinear MLP with elementwise worker/task interaction |
| Eq14–15 | assignment + update_fairness_memory | fixed validation threshold; hard allocations update utility |
| Eq16 | composite_loss | positive weight N-/N+, normalization Z=weight*N+ + N- |
| Eq17 | soft_allocation | task normalization `1+sum(p)` then worker capacity scaling |
| Eq18–22 | composite_loss | raw preference utility and normalized km / active task count; prior cumulative U in Gini |
| Eq23–28 | CandidateGraphMaterializer | bounded positive Eq8 deficit, train-profile similarity, km/dmax, slack, top-K |
| Eq33 | prepare.prepare | exact category count + global normalized frequency affinity, independent of learned scores |
| C.3, F | ReplayMetrics | fixed prediction samples; reward coverage; cumulative Gini; reward / opportunity Gini |

## Explicit choices not uniquely determined by the PDF

These are executable defaults, not recovered experimental settings or claims that the
original author code used these exact choices.

1. **Decision clock:** every release, completion, expiry or availability transition with
   pending tasks triggers one decision. Monthly grouping is not an allocation horizon.
   The EMA advances once per such decision, not once per accepted edge. Parameters are
   updated after each training decision; prior hard allocations are detached (no BPTT).
2. **Initial states and population:** all registered IDs across splits are known as roster
   metadata. Fairness uses this fixed population, including inactive workers. Every split
   starts at zero utility/memory and train-derived initial locations. No outcomes from
   validation are carried into test. A new worker uses global prior affinity, train-mean
   location, noon availability and capacity one. This is an explicit synthetic cold-start
   assumption, not inferred knowledge of future locations.
3. **Historical profiles:** train-period counts/coordinates are fitted once and frozen in
   all splits, including training replay. Thus training is supervised fitting with train
   statistics, not a strictly prefix-only historical reconstruction. Validation/test
   behaviors never fit these profiles. For strict prefix-only training experiments a
   separate warm-up/prefix feature protocol would be needed and should be reported.
4. **Availability:** circular mean time of day (UTC) gives the center of a 12-hour daily
   window. Capacity is ceil(mean interactions in-window per training calendar day),
   capped at 5. Last train location initializes each known worker; movement persists within
   a split. Arrival must precede both task deadline and window closing. Service duration
   is zero (unspecified in PDF). A worker has one simultaneous travel job; zero-distance
   assignments complete at a strictly later epsilon event to avoid duplicate scheduling.
5. **Rewards:** seeded U(0.5,1.5), generated once in preprocessing and reused across model
   seeds/variants. Affinity has no social enrichment; candidate retrieval uses a normalized
   own-plus-mean-neighbor historical profile. Unknown task categories have zero training
   frequency and hence zero affinity. Social edges with timestamps are filtered to the
   training cutoff; an undated edge file must represent the known training-time graph.
6. **Prediction evaluation:** fixed reference locations/availability at task release give
   common candidate samples. If the recorded worker is infeasible, that task has no
   supervised positive sample (and is still replayed); no positive is forced into the
   graph. Use at most two distinct feasible unobserved negatives per positive, seeded by
   task ID. Labels/pairs are fixed across epochs and model seeds, independent of top-K and
   simulated mobility. Reference worker feature coordinates remain fixed; fairness states
   used in the model follow each policy's actual allocation history. AUC/ACC describe
   this reference prediction benchmark, not acceptance/completion. Scores are not claimed
   to be calibrated probabilities under class weighting and auxiliary objectives.
7. **Loss edge sets:** BCE uses the fixed supervised reference edges. Allocation losses use
   retained dynamic feasible candidates, with threshold mask treated as stop-gradient.
   When a step has a single label class, BCE uses weight one; no labels gives differentiable
   zero prediction loss. This resolves the undefined N-/N+ boundary cases.
8. **Threshold/model selection:** best epoch by validation AUC (validation loss fallback if
   AUC undefined); then validation-only threshold sweep, maximizing ACC, with Utility and
   lower Gini as tie-breakers. No tuning on test. The paper does not specify this criterion.
9. **Additional defaults:** K=32, lambda_p=.7, dmax=60 km, retrieval weights and score-level
   coefficients are configurable engineering defaults; the PDF does not give their exact
   experiment values. Hidden size=64, heads=4, gamma=.9, lambda_f=1, Adam lr=.01, 500 epochs
   follow Appendix C.4 / the method. No new theorem or calibration claim is introduced.
10. **Opportunity fairness:** count all reference feasible worker/task opportunities once
    per released task, before pruning, independent of policy mobility. Divide cumulative
    preference utility by this count. Zero-opportunity workers are excluded from this
    secondary Gini and counted explicitly. Primary and reward Gini include all workers.
11. **Timing and scale:** Time covers candidate selection, model scoring and greedy
    allocation, with CUDA synchronization; excludes feature/feasible-graph construction,
    reference evaluation, loss and backward. No claim of matching Table 3 timing. Dynamic
    feasibility currently builds an M by active-N matrix, and social encoding covers the
    enrolled graph. Top-K reduces learned pair scoring, not the full retrieval cost.
    Reference samples use a bounded cache; window events are generated one day ahead.

## Validation boundaries

All equation/constraint tests and end-to-end toy runs are local CPU tests. Real processed
Foursquare/Yelp data, expert label judgments, exact candidate parameters and original
training logs were not supplied. No published numbers were embedded or generated.
New checkpoints cannot load 1.x parameters because gate/predictor shapes and semantics
changed. The retained legacy monthly adapter is deliberately labeled non-paper replay.
