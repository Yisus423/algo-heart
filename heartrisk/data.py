"""Load the raw release, validate it against the contract, and build matrices.

The functions here are deliberately explicit rather than vectorised. Row-at-a-time
validation costs about a second on the full 9000-row release, and in exchange every
model input is checked by name before it can reach a trained model.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pandas.api.types import is_numeric_dtype

from .schema import (
    FEATURE_ORDER,
    FIELDS,
    REQUIRED_RAW_COLUMNS,
    TARGET,
    Field,
    SchemaError,
    encode_binary_category,
    to_native,
    validate_value,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_PATH = REPO_ROOT / "data" / "raw" / "heart_disease_risk_2026.csv"
DEFAULT_CLEAN_PATH = REPO_ROOT / "data" / "processed" / "heart_clean.csv"

#: Column written alongside the model inputs so the cleaned file is auditable.
CLEAN_COLUMNS: tuple[str, ...] = tuple(field.name for field in FIELDS) + (TARGET,)


def load_raw(path: Path | str = DEFAULT_RAW_PATH) -> pd.DataFrame:
    """Read the raw release and confirm the contract's columns are present."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"raw dataset not found at {path}; run `python scripts/fetch_data.py`"
        )
    frame = pd.read_csv(path)
    require_columns(frame, REQUIRED_RAW_COLUMNS)
    return frame


def require_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    """Raise ``SchemaError`` naming every column the frame is missing."""
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise SchemaError(
            f"missing required columns: {missing}; received {list(frame.columns)}"
        )


def _native(value: object) -> object:
    """Local alias kept so this module reads the same as its siblings."""
    return to_native(value)


def validate_record(record: dict[str, object], *, require_target: bool = False) -> None:
    """Validate one record (a user payload or a dataset row) by field name."""
    for field in FIELDS:
        if field.name not in record:
            raise SchemaError(f"missing field {field.name!r}")
        validate_value(field, _native(record[field.name]))

    if require_target:
        if TARGET not in record:
            raise SchemaError(f"missing target column {TARGET!r}")
        target = _native(record[TARGET])
        if target not in (0, 1):
            raise SchemaError(f"{TARGET}: expected 0 or 1, got {target!r}")


def clean(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate the raw release and return the cleaned, model-ready frame.

    Differences from this project's original cleanup:

    * ``sex`` is encoded from its actual labels instead of ``1 if x == 2 else 0``,
      which silently produced an all-zero column.
    * Nothing else is dropped. The original script discarded ``st_depression``
      (the third strongest predictor at |r| = 0.36), ``hba1c`` and ``bmi`` while
      keeping weaker features; the measured effect of that choice was recorded in
      ``odd/tasks/heart-risk-chat-demo.md``.
    * ``patient_id`` is dropped because it identifies a row and carries no signal.
    """
    require_columns(frame, REQUIRED_RAW_COLUMNS)

    columns: dict[str, pd.Series] = {}
    for field in FIELDS:
        columns[field.name] = pd.Series(
            [_encode(field, value) for value in frame[field.name]],
            index=frame.index,
        )

    target = pd.Series(
        [_native(value) for value in frame[TARGET]], index=frame.index, name=TARGET
    )
    invalid = sorted({value for value in target if value not in (0, 1)})
    if invalid:
        raise SchemaError(f"{TARGET}: expected only 0 and 1, found {invalid}")

    cleaned = pd.DataFrame(columns, index=frame.index)
    cleaned[TARGET] = target
    return cleaned[list(CLEAN_COLUMNS)]


def _encode(field: Field, value: object) -> object:
    """Validate one raw value and return its encoded form."""
    native = _native(value)
    validate_value(field, native)
    if field.kind == "binary_category":
        return encode_binary_category(field, str(native))
    if field.kind == "boolean":
        return bool(native)
    return native


def to_features(cleaned: pd.DataFrame) -> pd.DataFrame:
    """Build the design matrix in the frozen ``FEATURE_ORDER``.

    Expects the output of :func:`clean`, where categorical fields are still
    labels. A categorical field expands to one column per declared category, so
    the layout never depends on which categories a batch happens to contain.

    Missing values are rejected here rather than imputed: imputation is a
    serving decision that must be declared to the user, and it lives in
    ``heartrisk.predict``.
    """
    require_columns(cleaned, tuple(field.name for field in FIELDS))

    columns: dict[str, object] = {}
    for field in FIELDS:
        series = cleaned[field.name]
        if series.isna().any():
            count = int(series.isna().sum())
            raise SchemaError(
                f"{field.name}: {count} missing value(s); impute before building features"
            )
        if field.kind == "categorical":
            for category in field.categories:
                columns[f"{field.name}={category}"] = (series == category).astype("float64")
        else:
            if not is_numeric_dtype(series):
                raise SchemaError(
                    f"{field.name}: expected an encoded numeric column but found "
                    f"dtype {series.dtype!r}; pass the output of clean()"
                )
            columns[field.name] = series.astype("float64")

    return pd.DataFrame(columns, index=cleaned.index)[list(FEATURE_ORDER)]


def write_clean(cleaned: pd.DataFrame, path: Path | str = DEFAULT_CLEAN_PATH) -> Path:
    """Write the cleaned frame to disk, creating the directory if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(path, index=False)
    return path
