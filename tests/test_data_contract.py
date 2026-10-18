"""Contract tests for the cleaned dataset and the frozen design matrix.

Every test here drives the public interface (`heartrisk.data` and
`heartrisk.schema`) and asserts observable effects: raised errors, column
identity, and the pinned shape of the regenerated dataset.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from heartrisk.data import clean, load_raw, to_features
from heartrisk.schema import FEATURE_ORDER, FIELDS, TARGET, SchemaError

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = REPO_ROOT / "data" / "raw" / "heart_disease_risk_2026.csv"


def valid_row(**overrides: object) -> dict[str, object]:
    """One plausible record covering every field in the contract."""
    row: dict[str, object] = {
        "age": 54,
        "sex": "Male",
        "resting_bp_systolic": 128,
        "resting_bp_diastolic": 81,
        "cholesterol_total": 200,
        "hdl": 55,
        "ldl": 103,
        "triglycerides": 152,
        "fasting_blood_sugar": 119,
        "hba1c": 5.4,
        "bmi": 26.1,
        "resting_heart_rate": 81,
        "max_heart_rate_achieved": 165,
        "chest_pain_type": "Asymptomatic",
        "exercise_induced_angina": False,
        "st_depression": 0.8,
        "family_history": False,
        "smoker_status": "Never",
        "alcohol_units_per_week": 3.0,
        "exercise_minutes_per_week": 139,
        "sleep_hours": 6.8,
        "stress_score": 42.0,
        "wearable_owner": True,
        "daily_steps": 6178,
        "diet_quality_score": 60.0,
        TARGET: 0,
    }
    row.update(overrides)
    return row


def valid_frame(rows: list[dict[str, object]] | None = None) -> pd.DataFrame:
    return pd.DataFrame(rows if rows is not None else [valid_row()])


# --- cleaning ---------------------------------------------------------------


def test_sex_is_encoded_from_the_source_labels() -> None:
    """The source stores Male/Female; the previous pipeline zeroed the column."""
    frame = valid_frame([valid_row(sex="Male"), valid_row(sex="Female")])

    cleaned = clean(frame)

    assert cleaned["sex"].tolist() == [1, 0]


def test_clean_keeps_every_contract_field() -> None:
    cleaned = clean(valid_frame())

    expected = {field.name for field in FIELDS} | {TARGET}
    assert expected <= set(cleaned.columns)


def test_clean_preserves_the_target_untouched() -> None:
    frame = valid_frame([valid_row(**{TARGET: 1}), valid_row(**{TARGET: 0})])

    assert clean(frame)[TARGET].tolist() == [1, 0]


# --- validation -------------------------------------------------------------


def test_missing_required_column_is_rejected_by_name() -> None:
    frame = valid_frame().drop(columns=["daily_steps"])

    with pytest.raises(SchemaError, match="daily_steps"):
        clean(frame)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("age", 200),
        ("age", -1),
        ("bmi", -3.0),
        ("resting_bp_systolic", 0),
        ("sleep_hours", 30.0),
        ("daily_steps", -50),
    ],
)
def test_implausible_value_is_rejected_by_name(field: str, value: object) -> None:
    frame = valid_frame([valid_row(**{field: value})])

    with pytest.raises(SchemaError, match=field):
        clean(frame)


def test_unknown_category_is_rejected_by_name() -> None:
    frame = valid_frame([valid_row(smoker_status="Sometimes")])

    with pytest.raises(SchemaError, match="smoker_status"):
        clean(frame)


def test_missing_value_is_rejected_rather_than_read_as_zero() -> None:
    """A null that reaches the matrix would silently become 0."""
    frame = valid_frame()
    frame.loc[0, "hdl"] = float("nan")

    with pytest.raises(SchemaError, match="hdl"):
        clean(frame)


def test_non_numeric_value_in_a_numeric_field_is_rejected() -> None:
    frame = valid_frame([valid_row(daily_steps="muchos")])

    with pytest.raises(SchemaError, match="daily_steps"):
        clean(frame)


def test_to_features_rejects_a_frame_without_the_contract_columns() -> None:
    frame = valid_frame().drop(columns=["hdl", "ldl"])

    with pytest.raises(SchemaError, match="hdl"):
        to_features(frame)


def test_to_features_refuses_a_frame_that_was_never_cleaned() -> None:
    """A raw frame still holds sex as "Male", which is not a matrix column."""
    with pytest.raises(SchemaError, match="sex"):
        to_features(valid_frame())


def test_load_raw_reports_a_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        load_raw(REPO_ROOT / "data" / "raw" / "does-not-exist.csv")


# --- frozen design matrix ---------------------------------------------------


def test_feature_order_matches_the_frozen_contract() -> None:
    features = to_features(clean(valid_frame()))

    assert list(features.columns) == list(FEATURE_ORDER)


def test_feature_order_is_identical_for_one_row_and_for_many() -> None:
    single = to_features(clean(valid_frame()))
    batch = to_features(clean(valid_frame([valid_row(), valid_row(age=61)])))

    assert list(single.columns) == list(batch.columns) == list(FEATURE_ORDER)


def test_a_missing_category_still_contributes_its_columns() -> None:
    """A batch without 'Former' smokers must not shift every later column."""
    frame = valid_frame([valid_row(smoker_status="Never"), valid_row(smoker_status="Current")])

    features = to_features(clean(frame))

    former_columns = [c for c in features.columns if c.startswith("smoker_status=")]
    assert len(former_columns) == 3
    assert features[former_columns].to_numpy().sum() == 2


def test_categorical_values_become_one_hot_columns() -> None:
    frame = valid_frame([valid_row(chest_pain_type="Typical Angina")])

    features = to_features(clean(frame))

    assert features["chest_pain_type=Typical Angina"].tolist() == [1]
    assert features["chest_pain_type=Asymptomatic"].tolist() == [0]


# --- pinned regeneration of the real dataset --------------------------------


@pytest.mark.skipif(not RAW_PATH.exists(), reason="run scripts/fetch_data.py first")
def test_real_dataset_regenerates_the_pinned_cleaned_frame() -> None:
    cleaned = clean(load_raw(RAW_PATH))

    assert len(cleaned) == 9000
    assert cleaned[TARGET].value_counts().to_dict() == {0: 6273, 1: 2727}
    assert set(cleaned["sex"].unique()) == {0, 1}, "sex must not be a dead column"


@pytest.mark.skipif(not RAW_PATH.exists(), reason="run scripts/fetch_data.py first")
def test_real_dataset_has_no_missing_values() -> None:
    assert int(clean(load_raw(RAW_PATH)).isna().sum().sum()) == 0
