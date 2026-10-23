"""experiments/run_pipeline.py: plan building and stage selection only.

No step is executed -- every step is an existing script that has its own
tests/run, so this only checks that the plan is the right shape and that
the module paths it names really exist.
"""

import importlib.util

import pytest

from experiments import run_pipeline as rp

STAGES = rp.build_stages(rp.MODELS, ("SA1", "NSW1"), (1, 2, 3, 4, 5))


def test_every_step_module_exists():
    for stage in STAGES:
        for step in stage.steps:
            assert importlib.util.find_spec(step.module) is not None, step.module


def test_stage_order_follows_the_readme():
    names = [s.name for s in STAGES]
    assert names == [
        "data", "synthetic", "detect", "forecast", "adapt",
        "uncertainty", "escalation", "produce",
    ]


def test_default_runs_cheap_stages_and_reports_heavy_ones_skipped():
    to_run, skipped = rp.select_stages(STAGES, None, include_heavy=False)
    assert [s.name for s in to_run] == ["data", "synthetic", "detect", "forecast", "produce"]
    assert {s.name for s in skipped} == {"adapt", "uncertainty", "escalation"}


def test_all_without_heavy_skips_heavy_stages():
    to_run, skipped = rp.select_stages(STAGES, ["all"], include_heavy=False)
    assert "adapt" not in [s.name for s in to_run]
    assert {"adapt", "uncertainty", "escalation"} <= {s.name for s in skipped}


def test_naming_a_heavy_stage_runs_it():
    to_run, _ = rp.select_stages(STAGES, ["adapt"], include_heavy=False)
    assert [s.name for s in to_run] == ["adapt"]


def test_stages_run_in_pipeline_order_whatever_the_user_types():
    to_run, _ = rp.select_stages(STAGES, ["produce", "detect"], include_heavy=False)
    assert [s.name for s in to_run] == ["detect", "produce"]


def test_unknown_stage_is_rejected():
    with pytest.raises(SystemExit):
        rp.select_stages(STAGES, ["nope"], include_heavy=False)


def test_adaptation_steps_seed_only_stochastic_models():
    steps = rp.adaptation_steps(("nhits", "dhr_arima"), ("SA1",), (1, 2))
    args = [" ".join(s.args) for s in steps]
    assert args == [
        "--model nhits --region SA1 --seed 1",
        "--model nhits --region SA1 --seed 2",
        "--model dhr_arima --region SA1",
    ]


def test_dry_run_runs_nothing(capsys, monkeypatch):
    monkeypatch.setattr(rp, "run_step", lambda s: pytest.fail("a step was executed"))
    assert rp.main(["--dry-run"]) == 0
    assert "Pipeline plan" in capsys.readouterr().out


def test_arms_option_is_passed_to_every_adaptation_step():
    steps = rp.adaptation_steps(("xgboost", "dhr_arima"), ("SA1",), (1,), arms="D_drift_recent_30d")
    assert steps and all(s.args[-2:] == ("--arms", "D_drift_recent_30d") for s in steps)


def test_tuning_sweep_only_with_retune():
    names = lambda stages: [  # noqa: E731
        s.module.rsplit(".", 1)[-1] for st in stages if st.name == "synthetic" for s in st.steps
    ]
    assert "run_fine_tune_on_synthetic" not in names(STAGES)
    retuned = rp.build_stages(rp.MODELS, ("SA1",), (1,), retune=True)
    assert "run_fine_tune_on_synthetic" in names(retuned)


def test_adapt_runs_both_detectors_then_rescores():
    adapt = next(s for s in STAGES if s.name == "adapt")
    detectors = {s.args[1] for s in adapt.steps if s.args and s.args[0] == "--detector"}
    assert detectors == {"adwin", "page_hinkley"}
    assert adapt.steps[-1].module.endswith("rescore_adaptation_runs")


def test_detectors_option_limits_the_adapt_stage():
    stages = rp.build_stages(("dhr_arima",), ("SA1",), (1,), detectors=("page_hinkley",))
    adapt = next(s for s in stages if s.name == "adapt")
    assert [s.args[:2] for s in adapt.steps[:-1]] == [("--detector", "page_hinkley")]


def test_calibration_forecasts_seed_only_stochastic_models():
    steps = rp.calibration_steps(("xgboost", "dhr_arima"), ("SA1",), (1, 2))
    assert [" ".join(s.args) for s in steps] == [
        "--model xgboost --region SA1 --seed 1",
        "--model xgboost --region SA1 --seed 2",
        "--model dhr_arima --region SA1",
    ]


def test_produce_stage_covers_every_table_and_figure():
    produce = next(s for s in STAGES if s.name == "produce")
    names = {s.module.rsplit(".", 1)[-1] for s in produce.steps}
    for t in ("t1_post_tune", "t3", "t4", "t5", "t6", "t7"):
        assert f"produce_table_{t}" in names or any(n.startswith(f"produce_table_{t}") for n in names)
    for f in ("f1", "f3", "f4", "f5", "f6"):
        assert f"produce_figure_{f}" in names
