"""Experiment layer: the only place that writes to results/.

`run_harness.record_run` is the one row-writer for results/runs.csv;
`run_*.py` scripts drive it, `produce_*.py` scripts group-by the ledger.
"""
