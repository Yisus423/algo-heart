"""Answer from a possibly incomplete payload, and declare what was missing.

This module exists because of the demo's central constraint. The interview can
only ask about habits, and the strongest predictors in this dataset are not
habits: ``max_heart_rate_achieved`` comes from a stress test, ``st_depression``
from an electrocardiogram, and the lipid panel from a blood draw. Measured on the
full release, a model restricted to interview-answerable fields reaches a recall
of roughly 0.49 against 0.73 for the full feature set.

So the serving layer never hides an imputation. Every answer carries the count of
real fields, the names of the fields that were filled in, the measurements the
user could still go and get, and a confidence level. Below ``MIN_COVERAGE`` it
returns no verdict at all, because the honest answer to a two-answer interview is
"this is not enough to tell you anything".
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

import pandas as pd

from .schema import (
    FEATURE_ORDER,
    FIELDS,
    FIELDS_BY_NAME,
    MEASUREMENT_FIELDS,
    SchemaError,
    derive_bmi,
    encode_binary_category,
    to_native,
    validate_value,
)
from .train import Artifact

#: Fraction of contract fields that must be real before a verdict is offered.
#: A full interview reaches 14 of 25 answers (0.56), which clears the floor by
#: design: a conversation on its own earns a verdict at the lowest confidence
#: level, never a confident one.
MIN_COVERAGE = 0.5

HIGH_CONFIDENCE_COVERAGE = 0.95
MEDIUM_CONFIDENCE_COVERAGE = 0.75

#: Interview values the app computes a field from, so the language model never
#: does arithmetic. Each entry is a plausibility bound.
DERIVED_MEASUREMENTS: dict[str, tuple[float, float]] = {
    "weight_kg": (2.0, 500.0),
    "height_cm": (30.0, 300.0),
}


class Confidence(str, Enum):
    """How much of the answer rests on real values rather than imputations."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True, slots=True)
class Prediction:
    """One answer, with everything needed to present it honestly.

    Attributes:
        probability: model output, always populated for diagnostics.
        label: 1 or 0 once ``estimate_available``, otherwise ``None``. When it is
            ``None`` the caller must not present ``probability`` as a risk
            figure; it is kept only so the UI can show a band.
        threshold: the artifact's cutoff, so the caller can explain the tradeoff.
        coverage: real fields divided by contract fields.
        confidence: band derived from ``coverage``.
        estimate_available: ``False`` below ``MIN_COVERAGE``.
        provided: contract fields that came from real values.
        imputed: contract fields filled in from the training distribution.
        missing_measurements: the subset of ``imputed`` that a doctor's visit
            would resolve, which is what the app tells the user to go get.
        out_of_distribution: provided values outside the training 1st-99th
            percentile range.
        schema_version: contract version the artifact was built with.
    """

    probability: float
    label: int | None
    threshold: float
    coverage: float
    confidence: Confidence
    estimate_available: bool
    provided: tuple[str, ...]
    imputed: tuple[str, ...]
    missing_measurements: tuple[str, ...]
    out_of_distribution: tuple[str, ...]
    schema_version: str

    @property
    def provided_count(self) -> int:
        return len(self.provided)

    @property
    def total_count(self) -> int:
        return len(FIELDS_BY_NAME)

    @property
    def is_out_of_distribution(self) -> bool:
        return bool(self.out_of_distribution)


def _confidence_for(coverage: float) -> Confidence:
    if coverage < MIN_COVERAGE:
        return Confidence.NONE
    if coverage >= HIGH_CONFIDENCE_COVERAGE:
        return Confidence.HIGH
    if coverage >= MEDIUM_CONFIDENCE_COVERAGE:
        return Confidence.MEDIUM
    return Confidence.LOW


def _as_number(key: str, value: object, minimum: float, maximum: float) -> float:
    native = to_native(value)
    if isinstance(native, bool) or not isinstance(native, (int, float)):
        raise SchemaError(f"{key}: expected a number, got {native!r}")
    if not minimum <= float(native) <= maximum:
        raise SchemaError(
            f"{key}: {native} is outside the plausible range {minimum}-{maximum}"
        )
    return float(native)


def resolve(payload: Mapping[str, object]) -> dict[str, object]:
    """Validate a partial payload and return the contract fields it supplies.

    Unknown keys are refused rather than ignored, because an extraction typo
    (``cholestrol``) would otherwise look like a field the user simply did not
    provide, and the answer would be built from a median instead of the value
    the user actually gave.
    """
    if not isinstance(payload, Mapping):
        raise SchemaError(f"payload must be a mapping, got {type(payload).__name__}")

    resolved: dict[str, object] = {}
    derived: dict[str, float] = {}
    for key, value in payload.items():
        if key in DERIVED_MEASUREMENTS:
            minimum, maximum = DERIVED_MEASUREMENTS[key]
            derived[key] = _as_number(key, value, minimum, maximum)
            continue

        contract_field = FIELDS_BY_NAME.get(key)
        if contract_field is None:
            raise SchemaError(
                f"unknown field {key!r}; expected one of {sorted(FIELDS_BY_NAME)}"
            )
        native = to_native(value)
        validate_value(contract_field, native)
        resolved[key] = native

    # BMI needs both measurements. A lone weight is ignored rather than guessed.
    if "bmi" not in resolved and set(derived) == set(DERIVED_MEASUREMENTS):
        resolved["bmi"] = derive_bmi(derived["weight_kg"], derived["height_cm"])

    return resolved


def design_row(resolved: Mapping[str, object], artifact: Artifact) -> pd.DataFrame:
    """Build a one-row design matrix in the artifact's exact column order."""
    if tuple(artifact.feature_order) != FEATURE_ORDER:
        raise SchemaError(
            "artifact feature layout does not match the contract; rebuild with "
            "`python -m heartrisk.train`"
        )

    columns: dict[str, float] = {}
    for contract_field in FIELDS:
        if contract_field.kind == "categorical":
            supplied = resolved.get(contract_field.name)
            for category in contract_field.categories:
                columns[f"{contract_field.name}={category}"] = (
                    1.0 if supplied == category else 0.0
                )
            continue

        if contract_field.name in resolved:
            supplied = resolved[contract_field.name]
            if contract_field.kind == "binary_category":
                # Same encoder clean() uses, so training and serving cannot drift.
                columns[contract_field.name] = float(
                    encode_binary_category(contract_field, str(supplied))
                )
            else:
                columns[contract_field.name] = float(supplied)  # type: ignore[arg-type]
        elif contract_field.name in artifact.imputation:
            columns[contract_field.name] = float(artifact.imputation[contract_field.name])
        else:
            raise SchemaError(
                f"artifact has no imputation value for {contract_field.name!r}; "
                "rebuild with `python -m heartrisk.train`"
            )

    return pd.DataFrame([columns])[list(artifact.feature_order)]


def predict(payload: Mapping[str, object], artifact: Artifact) -> Prediction:
    """Answer from a partial payload and declare how much of it was real."""
    resolved = resolve(payload)
    row = design_row(resolved, artifact)
    probability = float(artifact.model.predict_proba(row)[0, 1])

    provided = tuple(f.name for f in FIELDS if f.name in resolved)
    imputed = tuple(f.name for f in FIELDS if f.name not in resolved)
    coverage = len(provided) / len(FIELDS)
    confidence = _confidence_for(coverage)
    available = confidence is not Confidence.NONE

    out_of_distribution = tuple(
        f.name
        for f in FIELDS
        if f.kind == "numeric"
        and f.name in resolved
        and f.name in artifact.ood_bounds
        and not (
            artifact.ood_bounds[f.name][0]
            <= float(resolved[f.name])  # type: ignore[arg-type]
            <= artifact.ood_bounds[f.name][1]
        )
    )

    return Prediction(
        probability=probability,
        label=int(probability >= artifact.threshold) if available else None,
        threshold=artifact.threshold,
        coverage=coverage,
        confidence=confidence,
        estimate_available=available,
        provided=provided,
        imputed=imputed,
        missing_measurements=tuple(n for n in MEASUREMENT_FIELDS if n not in resolved),
        out_of_distribution=out_of_distribution,
        schema_version=artifact.schema_version,
    )
