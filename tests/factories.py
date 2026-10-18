"""Shared builders for test frames.

``valid_row`` produces one plausible record for every field in the contract, so a
test that wants to exercise one rule only has to override that one value.
``labelled_frame`` produces a frame with a learnable signal, which lets the
training tests run on a few hundred rows instead of the full release.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from heartrisk.schema import TARGET


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
    """A contract-satisfying frame. Defaults to a single row."""
    return pd.DataFrame(rows if rows is not None else [valid_row()])


def labelled_frame(n: int = 600, seed: int = 7) -> pd.DataFrame:
    """A frame whose target is driven by age and max heart rate.

    The signal is deliberately simple so a small random forest can learn it and
    the training tests stay fast.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for _ in range(n):
        age = int(rng.integers(20, 85))
        max_hr = int(rng.integers(95, 210))
        risk = (age - 20) / 65 * 0.7 + (210 - max_hr) / 115 * 0.7 + rng.normal(0, 0.15)
        rows.append(
            valid_row(age=age, max_heart_rate_achieved=max_hr, **{TARGET: int(risk > 0.7)})
        )
    return valid_frame(rows)
