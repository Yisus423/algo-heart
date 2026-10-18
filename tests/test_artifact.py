"""Tests for the frozen artifact and the serving layer.

These drive ``heartrisk.train`` and ``heartrisk.predict`` only through their
public functions and assert observable effects: the files written, the values
returned, and the errors raised.
"""

from __future__ import annotations

import dataclasses
import subprocess
import sys
from pathlib import Path

import pytest

from heartrisk import predict as predict_module
from heartrisk import train as train_module
from heartrisk.artifact import Artifact
from heartrisk.data import clean
from heartrisk.schema import (
    ASKABLE_FIELDS,
    FEATURE_ORDER,
    FIELDS_BY_NAME,
    SCHEMA_VERSION,
    TARGET,
    SchemaError,
)
from tests.factories import labelled_frame, valid_row

# --- threshold selection ----------------------------------------------------


def test_threshold_is_the_strictest_one_that_still_reaches_target_recall() -> None:
    y_true = [0, 0, 1, 1, 1, 0, 1, 1]
    proba = [0.05, 0.35, 0.40, 0.55, 0.65, 0.80, 0.85, 0.95]

    threshold = train_module.choose_threshold(y_true, proba, target_recall=0.8)

    predicted = [int(p >= threshold) for p in proba]
    true_positives = sum(p for p, t in zip(predicted, y_true) if t == 1)
    assert true_positives / sum(y_true) >= 0.8
    # Raising the bar by one notch would lose a true positive and miss the target.
    stricter = [int(p > threshold) for p in proba]
    assert sum(p for p, t in zip(stricter, y_true) if t == 1) / sum(y_true) < 0.8


def test_threshold_flags_everyone_when_no_cutoff_reaches_the_target() -> None:
    threshold = train_module.choose_threshold([0, 1], [0.9, 0.1], target_recall=1.0)

    assert all(p >= threshold for p in (0.9, 0.1))


# --- training ---------------------------------------------------------------


#: Small forest keeps the suite fast; the training path is identical.
TEST_TREES = 60


@pytest.fixture(scope="module")
def artifact():
    return train_module.fit(clean(labelled_frame()), n_estimators=TEST_TREES)


@pytest.fixture(scope="module")
def model(artifact, tmp_path_factory):
    path = tmp_path_factory.mktemp("artifact") / "model.joblib"
    train_module.save_artifact(artifact, path)
    return train_module.load_artifact(path)


def test_training_is_reproducible() -> None:
    first = train_module.fit(clean(labelled_frame(seed=11)), n_estimators=TEST_TREES)
    second = train_module.fit(clean(labelled_frame(seed=11)), n_estimators=TEST_TREES)

    assert first.metrics["ranking"] == second.metrics["ranking"]
    assert first.threshold == second.threshold


def test_artifact_records_the_contract_it_was_built_from(artifact) -> None:
    assert tuple(artifact.feature_order) == FEATURE_ORDER
    assert artifact.schema_version == SCHEMA_VERSION


def test_artifact_carries_an_imputation_value_for_every_numeric_field(artifact) -> None:
    numeric = [f.name for f in FIELDS_BY_NAME.values() if f.kind != "categorical"]

    assert set(numeric) <= set(artifact.imputation)


def test_artifact_reports_a_measured_interview_only_baseline(artifact) -> None:
    """The confidence band needs a measured 'chat alone' number, not a guess."""
    baseline = artifact.metrics["interview_only"]

    assert 0 < baseline["feature_count"] < len(FEATURE_ORDER)
    assert 0.0 <= baseline["recall"] <= 1.0
    assert 0.0 <= baseline["precision"] <= 1.0
    assert 0.0 <= baseline["pr_auc"] <= 1.0


def test_save_and_load_round_trip(tmp_path: Path, artifact) -> None:
    path = tmp_path / "model.joblib"

    train_module.save_artifact(artifact, path)
    loaded = train_module.load_artifact(path)

    assert loaded.feature_order == artifact.feature_order
    assert loaded.threshold == artifact.threshold


def test_a_stale_artifact_is_refused_at_load_time(tmp_path: Path, artifact) -> None:
    path = tmp_path / "model.joblib"
    stale = dataclasses.replace(artifact, schema_version="0.0.1-stale")
    train_module.save_artifact(stale, path)

    with pytest.raises(SchemaError, match="schema"):
        train_module.load_artifact(path)


def test_a_missing_artifact_names_the_training_command(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="heartrisk.train"):
        train_module.load_artifact(tmp_path / "nope.joblib")


def test_the_artifact_class_is_importable_and_not_main() -> None:
    """`python -m heartrisk.train` runs that module as __main__.

    A class defined there would be pickled as __main__.Artifact, and an app that
    imports the package normally could never unpickle it. Keeping the class in
    its own module is what makes the documented command produce a usable file.
    """
    assert Artifact.__module__ == "heartrisk.artifact"


def test_the_command_line_entry_point_produces_a_loadable_artifact(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    raw = tmp_path / "raw.csv"
    out = tmp_path / "model.joblib"
    labelled_frame(200, seed=3).to_csv(raw, index=False)

    completed = subprocess.run(
        [
            sys.executable, "-m", "heartrisk.train",
            "--raw", str(raw), "--out", str(out),
            "--metrics", str(tmp_path / "metrics.json"),
            "--trees", "20",
        ],
        cwd=repo_root, capture_output=True, text=True, check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert str(out) in completed.stdout
    reloaded = train_module.load_artifact(out)
    assert tuple(reloaded.feature_order) == FEATURE_ORDER


# --- serving ----------------------------------------------------------------


def test_predict_reports_every_field_it_had_to_fill_in(model) -> None:
    payload = {"age": 61, "sex": "Male", "daily_steps": 4000}

    result = predict_module.predict(payload, model)

    assert set(result.provided) == set(payload)
    assert set(result.imputed) == set(FIELDS_BY_NAME) - set(payload)
    assert result.provided_count == 3
    assert result.total_count == len(FIELDS_BY_NAME)


def test_predict_is_order_independent(model) -> None:
    ordered = predict_module.predict({"age": 61, "hdl": 40, "ldl": 150}, model)
    shuffled = predict_module.predict({"ldl": 150, "age": 61, "hdl": 40}, model)

    assert ordered.probability == shuffled.probability


def test_predict_lists_the_measurements_the_user_still_needs(model) -> None:
    result = predict_module.predict({"age": 61, "sex": "Male"}, model)

    assert "hdl" in result.missing_measurements
    assert "max_heart_rate_achieved" in result.missing_measurements
    assert "age" not in result.missing_measurements


def test_predict_rejects_an_impossible_value_instead_of_imputing(model) -> None:
    with pytest.raises(SchemaError, match="age"):
        predict_module.predict({"age": 500}, model)


def test_predict_rejects_an_unknown_field_name(model) -> None:
    with pytest.raises(SchemaError, match="cholestrol"):
        predict_module.predict({"cholestrol": 200}, model)


def test_predict_rejects_an_unknown_category(model) -> None:
    with pytest.raises(SchemaError, match="smoker_status"):
        predict_module.predict({"smoker_status": "Sometimes"}, model)


def test_predict_flags_a_value_outside_the_training_distribution(model) -> None:
    result = predict_module.predict({"daily_steps": 99_000}, model)

    assert "daily_steps" in result.out_of_distribution


def test_predict_accepts_height_and_weight_and_derives_bmi(model) -> None:
    result = predict_module.predict({"weight_kg": 82.0, "height_cm": 175.0}, model)

    assert "bmi" in result.provided
    assert "bmi" not in result.imputed


# --- coverage, the honest-confidence contract -------------------------------


def test_predict_refuses_a_verdict_below_the_coverage_floor(model) -> None:
    result = predict_module.predict({"age": 61, "sex": "Male"}, model)

    assert result.estimate_available is False
    assert result.label is None


def test_a_complete_interview_earns_a_verdict_at_the_low_confidence_end(model) -> None:
    """The interview-answerable fields alone are 14 of 25, just over the floor."""
    payload = {name: valid_row()[name] for name in ASKABLE_FIELDS}

    result = predict_module.predict(payload, model)

    assert result.provided_count == len(ASKABLE_FIELDS)
    assert result.estimate_available is True
    assert result.confidence == predict_module.Confidence.LOW


def test_a_full_record_earns_the_highest_confidence(model) -> None:
    payload = {k: v for k, v in valid_row().items() if k != TARGET}

    result = predict_module.predict(payload, model)

    assert result.provided_count == len(FIELDS_BY_NAME)
    assert result.coverage == 1.0
    assert result.confidence == predict_module.Confidence.HIGH
    assert result.imputed == ()
