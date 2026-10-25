# Verification and remaining author inputs

Run: `python -m pytest -q`, `ruff check src scripts tests`, and README toy commands.
See `VALIDATION.json` for checks executed on this delivered version.

Before reporting new paper results, supply the actual processed records, expert-retained
labels and known-time social graph; choose/record unspecified protocol parameters; run
all baselines under common candidate sampling and reward seeds; and rerun repeated model
seeds. Metrics in toy validation are synthetic and must not be used as benchmark results.
