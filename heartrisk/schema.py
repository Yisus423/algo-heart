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
        source_es: short heading for grouping the field by where it comes from,
            e.g. "Análisis de sangre". ``needs_es`` is a phrase for a sentence
            ("Sale de un análisis de sangre"), so using it as a heading read as a
            fragment: "un tensiómetro — ○ Presión sistólica en reposo". It is part
            of the contract so that a new field cannot ship without one.
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
    source_es: str | None = None
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
        question_es="¿Cuántos años tienes?",
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
        question_es="¿Tienes tu presión sistólica medida en reposo?",
        askable=False,
        needs_es="un tensiómetro",
        source_es="Tensiómetro",
        unit_es="mmHg",
        minimum=50,
        maximum=300,
    ),
    Field(
        name="resting_bp_diastolic",
        kind="numeric",
        label_es="Presión diastólica en reposo",
        question_es="¿Tienes tu presión diastólica medida en reposo?",
        askable=False,
        needs_es="un tensiómetro",
        source_es="Tensiómetro",
        unit_es="mmHg",
        minimum=20,
        maximum=200,
    ),
    Field(
        name="cholesterol_total",
        kind="numeric",
        label_es="Colesterol total",
        question_es="¿Tienes tu colesterol total de un análisis reciente?",
        askable=False,
        needs_es="un análisis de sangre",
        source_es="Análisis de sangre",
        unit_es="mg/dL",
        minimum=50,
        maximum=700,
    ),
    Field(
        name="hdl",
        kind="numeric",
        label_es="Colesterol HDL",
        question_es="¿Tienes tu HDL de un análisis reciente?",
        askable=False,
        needs_es="un análisis de sangre",
        source_es="Análisis de sangre",
        unit_es="mg/dL",
        minimum=5,
        maximum=200,
    ),
    Field(
        name="ldl",
        kind="numeric",
        label_es="Colesterol LDL",
        question_es="¿Tienes tu LDL de un análisis reciente?",
        askable=False,
        needs_es="un análisis de sangre",
        source_es="Análisis de sangre",
        unit_es="mg/dL",
        minimum=10,
        maximum=500,
    ),
    Field(
        name="triglycerides",
        kind="numeric",
        label_es="Triglicéridos",
        question_es="¿Tienes tus triglicéridos de un análisis reciente?",
        askable=False,
        needs_es="un análisis de sangre",
        source_es="Análisis de sangre",
        unit_es="mg/dL",
        minimum=10,
        maximum=2000,
    ),
    Field(
        name="fasting_blood_sugar",
        kind="numeric",
        label_es="Glucemia en ayunas",
        question_es="¿Tienes tu glucemia en ayunas de un análisis reciente?",
        askable=False,
        needs_es="un análisis de sangre",
        source_es="Análisis de sangre",
        unit_es="mg/dL",
        minimum=20,
        maximum=600,
    ),
    Field(
        name="hba1c",
        kind="numeric",
        label_es="Hemoglobina glicosilada",
        question_es="¿Tienes tu hemoglobina glicosilada de un análisis reciente?",
        askable=False,
        needs_es="un análisis de sangre",
        source_es="Análisis de sangre",
        unit_es="%",
        minimum=2,
        maximum=20,
    ),
    Field(
        name="bmi",
        kind="numeric",
        label_es="Índice de masa corporal",
        question_es="¿Cuánto pesas y cuánto mides?",
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
        question_es="¿Sabes cuántas pulsaciones por minuto tienes en reposo?",
        askable=False,
        needs_es="un tensiómetro, un oxímetro o tu reloj",
        source_es="Pulso en reposo",
        unit_es="lpm",
        minimum=20,
        maximum=250,
    ),
    Field(
        name="max_heart_rate_achieved",
        kind="numeric",
        label_es="Frecuencia cardíaca máxima alcanzada",
        question_es="¿Tienes el valor de frecuencia máxima de una prueba de esfuerzo?",
        askable=False,
        needs_es="una prueba de esfuerzo",
        source_es="Prueba de esfuerzo",
        unit_es="lpm",
        minimum=40,
        maximum=250,
    ),
    Field(
        name="chest_pain_type",
        kind="categorical",
        label_es="Tipo de dolor de pecho",
        question_es=(
            "¿Sientes dolor o molestia en el pecho? ¿Aparece cuando haces esfuerzo, "
            "en reposo, o no lo sientes nunca?"
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
        question_es="¿Sientes dolor o presión en el pecho cuando haces esfuerzo físico?",
        askable=True,
    ),
    Field(
        name="st_depression",
        kind="numeric",
        label_es="Depresión del segmento ST",
        question_es="¿Tienes el valor de depresión del segmento ST de un electrocardiograma?",
        askable=False,
        needs_es="un electrocardiograma",
        source_es="Electrocardiograma",
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
        question_es="¿Fumas, fumaste antes, o nunca has fumado?",
        askable=True,
        categories=("Never", "Former", "Current"),
    ),
    Field(
        name="alcohol_units_per_week",
        kind="numeric",
        label_es="Alcohol por semana",
        question_es="¿Cuántas bebidas alcohólicas tomas en una semana típica?",
        askable=True,
        unit_es="unidades",
        minimum=0,
        maximum=200,
    ),
    Field(
        name="exercise_minutes_per_week",
        kind="numeric",
        label_es="Minutos de ejercicio por semana",
        question_es="¿Cuántos minutos por semana haces actividad física que te haga sudar?",
        askable=True,
        unit_es="minutos",
        minimum=0,
        maximum=3000,
    ),
    Field(
        name="sleep_hours",
        kind="numeric",
        label_es="Horas de sueño",
        question_es="¿Cuántas horas duermes en una noche típica?",
        askable=True,
        unit_es="horas",
        minimum=0,
        maximum=24,
    ),
    Field(
        name="stress_score",
        kind="numeric",
        label_es="Estrés percibido",
        question_es="En una escala del 0 al 100, ¿cuánto estrés sientes en tu vida diaria?",
        askable=True,
        unit_es="0-100",
        minimum=0,
        maximum=100,
    ),
    Field(
        name="wearable_owner",
        kind="boolean",
        label_es="Usa reloj o banda de actividad",
        question_es="¿Usas reloj inteligente o banda de actividad?",
        askable=True,
    ),
    Field(
        name="daily_steps",
        kind="numeric",
        label_es="Pasos diarios",
        question_es="¿Cuántos pasos caminas por día, más o menos?",
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


#: Fields ordered by how much each one contributes to the model, measured on the
#: full release with a 300-tree forest's feature importances and read as a top-k
#: curve at the artifact's recall target:
#:
#:     top 4    recall 0.850   precision 0.616   PR-AUC 0.825
#:     top 8    recall 0.850   precision 0.719   PR-AUC 0.880
#:     all 25   recall 0.852   precision 0.734   PR-AUC 0.887
#:
#: Eight fields carry 98% of what twenty-five do. Three of them are close to
#: worthless *as questions* — ``sex`` 0.61%, ``family_history`` 0.48%,
#: ``wearable_owner`` 0.35% — which is why asking whether someone owns a smartwatch
#: was buying a third of a percent with a whole turn. A test fails if a field is
#: added to the contract without a place in this order.
IMPORTANCE_ORDER: tuple[str, ...] = (
    "max_heart_rate_achieved",
    "exercise_induced_angina",
    "st_depression",
    "age",
    "ldl",
    "hdl",
    "bmi",
    "exercise_minutes_per_week",
    "resting_bp_systolic",
    "hba1c",
    "fasting_blood_sugar",
    "cholesterol_total",
    "diet_quality_score",
    "stress_score",
    "daily_steps",
    "triglycerides",
    "resting_heart_rate",
    "alcohol_units_per_week",
    "sleep_hours",
    "resting_bp_diastolic",
    "chest_pain_type",
    "smoker_status",
    "sex",
    "family_history",
    "wearable_owner",
)

#: The fields the interview actually chases. The rest are welcome and still reach
#: the model, which keeps all thirty columns, but the interview does not demand
#: them: seventeen more answers to gain 0.015 of precision is a questionnaire.
CORE_FIELDS: tuple[str, ...] = IMPORTANCE_ORDER[:8]


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


#: Interview values the app turns into a contract field, so the model never does
#: arithmetic. They are declared here rather than in the serving layer because the
#: tool schema is built from the contract, and a field the model cannot report is a
#: field it silently drops.
DERIVED_MEASUREMENTS: dict[str, tuple[float, float]] = {
    "weight_kg": (2.0, 500.0),
    "height_cm": (30.0, 300.0),
}


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


def rejection_message(field: Field, value: object) -> str:
    """Neutral Spanish rendering of why a value was refused, written for the person.

    ``SchemaError`` messages are English because the pipeline developer reads them
    and the tests assert on them. They must not reach the screen: echoing one
    produced "No pude usar alguno de esos valores: cholesterol_total: 900 is above
    the plausible maximum 700", half Spanish and half developer.
    """
    is_number = isinstance(value, Real) and not isinstance(value, bool)
    if is_number and field.minimum is not None and field.maximum is not None:
        if not field.minimum <= float(value) <= field.maximum:
            unit = f" {field.unit_es}" if field.unit_es else ""
            return (
                f"{field.label_es}: {value:g}{unit} está fuera del rango razonable "
                f"({field.minimum:g}-{field.maximum:g}{unit})."
            )
    # No trailing "no lo guardé" here: the caller's sentence already says it, and
    # repeating it produced "... así que no los guardé: X: fuera de rango, así que
    # no lo guardé."
    return f"{field.label_es}: no pude usar ese valor."


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


#: How each group of numbers is actually obtained, in neutral Spanish and without
#: prescribing anything. The professor asked for "que el modelo recomiende o de
#: instrucciones para obtener algunos datos", and the honest version of that is how
#: the value is produced, not what someone should go and do: which studies a person
#: needs is a doctor's call, and the app says so.
HOW_TO_GET_ES: dict[str, str] = {
    "Tensiómetro": (
        "La presión te la toman en dos minutos en cualquier farmacia o centro de "
        "salud, y un tensiómetro de brazo en casa sirve igual. Conviene medirla "
        "sentado y después de unos minutos de reposo."
    ),
    "Pulso en reposo": (
        "Se cuenta con dos dedos en la muñeca o el cuello durante 30 segundos y se "
        "multiplica por dos. Muchos relojes y tensiómetros lo muestran directamente."
    ),
    "Análisis de sangre": (
        "Pídele a tu médico un perfil lipídico: es un análisis de sangre en ayunas "
        "que trae el colesterol total, el HDL, el LDL y los triglicéridos juntos. La "
        "glucemia y la hemoglobina glicosilada suelen pedirse en el mismo análisis."
    ),
    "Prueba de esfuerzo": (
        "Ese valor sale de una prueba de esfuerzo, que es un estudio con indicación "
        "médica y no algo para hacerse por cuenta propia. Si no la tienes, el modelo "
        "completa el dato con un promedio del dataset y la estimación pierde fuerza."
    ),
    "Electrocardiograma": (
        "Sale de un electrocardiograma: un estudio de minutos e indoloro. El valor "
        "que usamos es el que el informe llama depresión del segmento ST."
    ),
}

#: Shown with the list of missing numbers, because a list of studies with no framing
#: reads like a prescription.
NOT_MEDICAL_ADVICE_ES = (
    "Nada de esto es una indicación médica: qué te conviene medir lo decide tu "
    "médico, y esta lista es sólo lo que a este modelo le falta."
)


def how_to_get(field: Field) -> str | None:
    """How the field's number is obtained, or ``None`` for what a person just knows."""
    if field.askable:
        return None
    return HOW_TO_GET_ES.get(field.source_es or "")


def source_group(field: Field) -> str:
    """Heading the field belongs under: where its number comes from.

    Falls back to the field's own label rather than to nothing, so a future
    measurement field that forgets ``source_es`` still shows up in the checklist
    instead of disappearing from it.
    """
    return field.source_es or field.label_es
