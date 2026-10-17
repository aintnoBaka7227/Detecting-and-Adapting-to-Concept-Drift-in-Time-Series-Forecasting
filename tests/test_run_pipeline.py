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


def test_stage_order_is_data_detect_adapt_produce():
    names = [s.name for s in STAGES]
    assert names.index("forecast") < names.index("detect") < names.index("adapt")
    assert names.index("adapt") < names.index("uncertainty") < names.index("produce")


def test_default_runs_cheap_stages_and_reports_heavy_ones_skipped():
    to_run, skipped = rp.select_stages(STAGES, None, include_heavy=False)
    assert [s.name for s in to_run] == ["forecast", "detect", "produce"]
    assert {s.name for s in skipped} == {"synthetic", "adapt"}


def test_all_without_heavy_skips_heavy_stages():
    to_run, skipped = rp.select_stages(STAGES, ["all"], include_heavy=False)
    assert "adapt" not in [s.name for s in to_run]
    assert {"synthetic", "adapt", "uncertainty"} <= {s.name for s in skipped}


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
