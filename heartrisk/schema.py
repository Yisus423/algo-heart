"""The data contract for the heart risk demo.

This module is the single source of truth shared by the training pipeline, the
serving code, and the LLM interview. Three things live here:

1. ``Field`` definitions for every model input, including whether a layperson
   can answer the question at all (``askable``) and what they would have to go
   get measured (``needs_es``).
2. Validation bounds. These are *plausibility* bounds, not distribution bounds:
   they catch values that cannot be true (a negative age, a blood pressure of
   zero) without rejecting unusual but real measurements.
3. ``FEATURE_ORDER``, the frozen column layout of the design matrix. It is
   derived from ``FIELDS`` so it can never depend on which categories happen to
   appear in a particular batch.

That last point is the whole reason this module exists. One-hot encoding a
categorical column with ``pandas.get_dummies`` produces a column set that
changes with the data. A training frame that happens to contain "Former" smokers
and a serving frame that does not would silently disagree about what column 19
means, and the model would keep answering with full confidence while reading the
wrong numbers. Building the matrix from an explicit contract removes that
failure mode.

This project's earlier cleanup mapped ``sex`` with ``1 if x == 2 else 0``. The
source column holds the strings ``"Male"`` and ``"Female"``, so all 9000 rows
became ``0`` and the feature was dead weight. ``binary_category`` encodes it from
the declared labels instead.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Literal

#: Name of the binary target. 1 means the synthetic patient carries a heart
#: disease label; 0 means they do not.
TARGET = "has_heart_disease"

#: Row identifier in the raw release. Never a model input.
IDENTIFIER = "patient_id"

#: Bump when a field is added, removed, or re-encoded. The trained artifact
#: records the version it was built with so a stale artifact cannot be served
#: against a newer contract.
SCHEMA_VERSION = "1.0.0"

Kind = Literal["numeric", "categorical", "boolean", "binary_category"]


class SchemaError(ValueError):
    """Raised when data does not satisfy the contract."""


def to_native(value: object) -> object:
    """Convert a numpy or pandas scalar into the plain Python value it wraps.

    Validation reasons about ``numbers.Real`` and ``str``. A ``numpy.int64`` is
    not a Python ``int``, and ``numpy.bool_`` is not a Python ``bool``, so every
    scalar is unwrapped before it is checked.
    """
    item = getattr(value, "item", None)
    return item() if callable(item) else value


@dataclass(frozen=True, slots=True)
class Field:
    """One model input, with everything the interview needs to ask for it.

    Args:
        name: column name in the raw release.
        kind: how the value is validated and encoded.
        label_es: human label shown in the UI.
        question_es: what the interviewer asks.
        askable: ``True`` when a person can answer from memory during a chat.
            ``False`` means the value only exists after a measurement.
        needs_es: for ``askable=False``, what the person would have to obtain.
            The app surfaces this when coverage is too low to answer.
        unit_es: unit shown next to the value, when there is one.
        minimum: smallest plausibly true value. Inclusive.
        maximum: largest plausibly true value. Inclusive.
        categories: allowed values for ``categorical`` and ``binary_category``.
            For ``binary_category`` the order defines the encoding: the first
            entry becomes 0 and the second becomes 1.
        computed_from: names of the values the interview actually collects. The
            app computes the field instead of letting the language model do
            arithmetic.
    """

    name: str
    kind: Kind
    label_es: str
    question_es: str
    askable: bool
    needs_es: str | None = None
    unit_es: str | None = None
    minimum: float | None = None
    maximum: float | None = None
    categories: tuple[str, ...] = ()
    computed_from: tuple[str, ...] = ()


FIELDS: tuple[Field, ...] = (
    Field(
        name="age",
        kind="numeric",
        label_es="Edad",
        question_es="¿Cuántos años tenés?",
        askable=True,
        unit_es="años",
        minimum=1,
        maximum=120,
    ),
    Field(
        name="sex",
        kind="binary_category",
        label_es="Sexo",
        question_es="¿Cuál es tu sexo?",
        askable=True,
        categories=("Female", "Male"),
    ),
    Field(
        name="resting_bp_systolic",
        kind="numeric",
        label_es="Presión sistólica en reposo",
        question_es="¿Tenés tu presión sistólica medida en reposo?",
        askable=False,
        needs_es="un tensiómetro",
        unit_es="mmHg",
        minimum=50,
        maximum=300,
    ),
    Field(
        name="resting_bp_diastolic",
        kind="numeric",
        label_es="Presión diastólica en reposo",
        question_es="¿Tenés tu presión diastólica medida en reposo?",
        askable=False,
        needs_es="un tensiómetro",
        unit_es="mmHg",
        minimum=20,
        maximum=200,
    ),
    Field(
        name="cholesterol_total",
        kind="numeric",
        label_es="Colesterol total",
        question_es="¿Tenés tu colesterol total de un análisis reciente?",
        askable=False,
        needs_es="un análisis de sangre",
        unit_es="mg/dL",
        minimum=50,
        maximum=700,
    ),
    Field(
        name="hdl",
        kind="numeric",
        label_es="Colesterol HDL",
        question_es="¿Tenés tu HDL de un análisis reciente?",
        askable=False,
        needs_es="un análisis de sangre",
        unit_es="mg/dL",
        minimum=5,
        maximum=200,
    ),
    Field(
        name="ldl",
        kind="numeric",
        label_es="Colesterol LDL",
        question_es="¿Tenés tu LDL de un análisis reciente?",
        askable=False,
        needs_es="un análisis de sangre",
        unit_es="mg/dL",
        minimum=10,
        maximum=500,
    ),
    Field(
        name="triglycerides",
        kind="numeric",
        label_es="Triglicéridos",
        question_es="¿Tenés tus triglicéridos de un análisis reciente?",
        askable=False,
        needs_es="un análisis de sangre",
        unit_es="mg/dL",
        minimum=10,
        maximum=2000,
    ),
    Field(
        name="fasting_blood_sugar",
        kind="numeric",
        label_es="Glucemia en ayunas",
        question_es="¿Tenés tu glucemia en ayunas de un análisis reciente?",
        askable=False,
        needs_es="un análisis de sangre",
        unit_es="mg/dL",
        minimum=20,
        maximum=600,
    ),
    Field(
        name="hba1c",
        kind="numeric",
        label_es="Hemoglobina glicosilada",
        question_es="¿Tenés tu hemoglobina glicosilada de un análisis reciente?",
        askable=False,
        needs_es="un análisis de sangre",
        unit_es="%",
        minimum=2,
        maximum=20,
    ),
    Field(
        name="bmi",
        kind="numeric",
        label_es="Índice de masa corporal",
        question_es="¿Cuánto pesás y cuánto medís?",
        askable=True,
        unit_es="kg/m²",
        minimum=8,
        maximum=80,
        computed_from=("weight_kg", "height_cm"),
    ),
    Field(
        name="resting_heart_rate",
        kind="numeric",
        label_es="Frecuencia cardíaca en reposo",
        question_es="¿Sabés cuántas pulsaciones por minuto tenés en reposo?",
        askable=False,
        needs_es="un tensiómetro, un oxímetro o tu reloj",
        unit_es="lpm",
        minimum=20,
        maximum=250,
    ),
    Field(
        name="max_heart_rate_achieved",
        kind="numeric",
        label_es="Frecuencia cardíaca máxima alcanzada",
        question_es="¿Tenés el valor de frecuencia máxima de una prueba de esfuerzo?",
        askable=False,
        needs_es="una prueba de esfuerzo",
        unit_es="lpm",
        minimum=40,
        maximum=250,
    ),
    Field(
        name="chest_pain_type",
        kind="categorical",
        label_es="Tipo de dolor de pecho",
        question_es=(
            "¿Sentís dolor o molestia en el pecho? ¿Aparece cuando hacés esfuerzo, "
            "en reposo, o no lo sentís nunca?"
        ),
        askable=True,
        categories=(
            "Asymptomatic",
            "Atypical Angina",
            "Non-Anginal Pain",
            "Typical Angina",
        ),
    ),
    Field(
        name="exercise_induced_angina",
        kind="boolean",
        label_es="Angina de esfuerzo",
        question_es="¿Sentís dolor o presión en el pecho cuando hacés esfuerzo físico?",
        askable=True,
    ),
    Field(
        name="st_depression",
        kind="numeric",
        label_es="Depresión del segmento ST",
        question_es="¿Tenés el valor de depresión del segmento ST de un electrocardiograma?",
        askable=False,
        needs_es="un electrocardiograma",
        unit_es="mm",
        minimum=-5,
        maximum=15,
    ),
    Field(
        name="family_history",
        kind="boolean",
        label_es="Antecedentes familiares",
        question_es=(
            "¿Alguien de tu familia directa tuvo problemas cardíacos antes de los 65 años?"
        ),
        askable=True,
    ),
    Field(
        name="smoker_status",
        kind="categorical",
        label_es="Tabaquismo",
        question_es="¿Fumás, fumaste antes, o nunca fumaste?",
        askable=True,
        categories=("Never", "Former", "Current"),
    ),
    Field(
        name="alcohol_units_per_week",
        kind="numeric",
        label_es="Alcohol por semana",
        question_es="¿Cuántas bebidas alcohólicas tomás en una semana típica?",
        askable=True,
        unit_es="unidades",
        minimum=0,
        maximum=200,
    ),
    Field(
        name="exercise_minutes_per_week",
        kind="numeric",
        label_es="Minutos de ejercicio por semana",
        question_es="¿Cuántos minutos por semana hacés actividad física que te haga sudar?",
        askable=True,
        unit_es="minutos",
        minimum=0,
        maximum=3000,
    ),
    Field(
        name="sleep_hours",
        kind="numeric",
        label_es="Horas de sueño",
        question_es="¿Cuántas horas dormís en una noche típica?",
        askable=True,
        unit_es="horas",
        minimum=0,
        maximum=24,
    ),
    Field(
        name="stress_score",
        kind="numeric",
        label_es="Estrés percibido",
        question_es="En una escala del 0 al 100, ¿cuánto estrés sentís en tu vida diaria?",
        askable=True,
        unit_es="0-100",
        minimum=0,
        maximum=100,
    ),
    Field(
        name="wearable_owner",
        kind="boolean",
        label_es="Usa reloj o banda de actividad",
        question_es="¿Usás reloj inteligente o banda de actividad?",
        askable=True,
    ),
    Field(
        name="daily_steps",
        kind="numeric",
        label_es="Pasos diarios",
        question_es="¿Cuántos pasos caminás por día, más o menos?",
        askable=True,
        unit_es="pasos",
        minimum=0,
        maximum=100_000,
    ),
    Field(
        name="diet_quality_score",
        kind="numeric",
        label_es="Calidad de la alimentación",
        question_es=(
            "En una escala del 0 al 100, ¿cómo calificarías tu alimentación? "
            "0 es comida ultraprocesada todos los días y 100 es vegetales, "
            "legumbres y poca comida procesada."
        ),
        askable=True,
        unit_es="0-100",
        minimum=0,
        maximum=100,
    ),
)

FIELDS_BY_NAME: dict[str, Field] = {field.name: field for field in FIELDS}

#: Every column the raw release must provide before cleanup runs.
REQUIRED_RAW_COLUMNS: tuple[str, ...] = tuple(field.name for field in FIELDS) + (TARGET,)

#: Fields a person can answer in conversation. The rest only exist after a
#: measurement, which is what the app has to tell the user when it cannot reach
#: a confident answer.
ASKABLE_FIELDS: tuple[str, ...] = tuple(f.name for f in FIELDS if f.askable)

#: Fields that require a device, a study, or a lab result.
MEASUREMENT_FIELDS: tuple[str, ...] = tuple(f.name for f in FIELDS if not f.askable)


def _columns_for(field: Field) -> tuple[str, ...]:
    if field.kind == "categorical":
        return tuple(f"{field.name}={category}" for category in field.categories)
    return (field.name,)


#: Frozen design-matrix layout. Derived from FIELDS, never from the data.
FEATURE_ORDER: tuple[str, ...] = tuple(
    column for field in FIELDS for column in _columns_for(field)
)

#: Spanish rendering of a design-matrix column, for the UI.
FEATURE_LABELS_ES: dict[str, str] = {
    column: (
        field.label_es
        if field.kind != "categorical"
        else f"{field.label_es}: {column.split('=', 1)[1]}"
    )
    for field in FIELDS
    for column in _columns_for(field)
}


def encode_binary_category(field: Field, value: str) -> int:
    """Map a declared label to 0 or 1 using the order in ``field.categories``."""
    if field.kind != "binary_category":
        raise SchemaError(f"{field.name} is not a binary_category field")
    if value not in field.categories:
        raise SchemaError(
            f"{field.name}: unknown value {value!r}; "
            f"expected one of {list(field.categories)}"
        )
    return field.categories.index(value)


def derive_bmi(weight_kg: float, height_cm: float) -> float:
    """Return BMI from the two values the interview actually collects.

    The interview asks for weight and height because both are exact answers a
    person can give, and computing BMI in Python removes a class of language
    model arithmetic errors.
    """
    if height_cm <= 0:
        raise SchemaError(f"height_cm must be positive, got {height_cm}")
    if weight_kg <= 0:
        raise SchemaError(f"weight_kg must be positive, got {weight_kg}")
    return weight_kg / (height_cm / 100) ** 2


def validate_value(field: Field, value: object) -> None:
    """Raise ``SchemaError`` unless ``value`` is acceptable for ``field``.

    Bounds are plausibility bounds: they reject impossible values without
    rejecting unusual measurements. Missing values are rejected too: a null
    that reaches the design matrix is a silent zero, not a missing value.
    """
    if field.kind in ("numeric",):
        # ``bool`` is a subclass of ``int``, so it has to be rejected first.
        if isinstance(value, bool) or not isinstance(value, Real):
            raise SchemaError(f"{field.name}: expected a number, got {value!r}")
        if math.isnan(float(value)):
            raise SchemaError(
                f"{field.name}: missing value; every model input needs a real measurement"
            )
        if field.minimum is not None and value < field.minimum:
            raise SchemaError(
                f"{field.name}: {value} is below the plausible minimum {field.minimum}"
            )
        if field.maximum is not None and value > field.maximum:
            raise SchemaError(
                f"{field.name}: {value} is above the plausible maximum {field.maximum}"
            )
        return

    if field.kind == "boolean":
        if not isinstance(value, bool):
            raise SchemaError(f"{field.name}: expected a boolean, got {value!r}")
        return

    if field.kind == "binary_category":
        if not isinstance(value, str) or value not in field.categories:
            raise SchemaError(
                f"{field.name}: unknown value {value!r}; "
                f"expected one of {list(field.categories)}"
            )
        return

    if field.kind == "categorical":
        if not isinstance(value, str) or value not in field.categories:
            raise SchemaError(
                f"{field.name}: unknown value {value!r}; "
                f"expected one of {list(field.categories)}"
            )
        return

    raise SchemaError(f"{field.name}: unsupported kind {field.kind!r}")
