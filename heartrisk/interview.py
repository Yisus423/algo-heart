"""The interview: turning a conversation into validated contract fields.

The seam
--------
An interviewer turns free text into values for :mod:`heartrisk.schema` fields.
Two implementations exist or will exist:

* :class:`StubInterviewer`, a deterministic question-and-parse loop. It needs no
  network and no API key, so the app always runs.
* A language-model interviewer (a later task) that conducts a natural
  conversation and emits its result as a tool call.

Both satisfy :class:`Interviewer`, and ``app.py`` only ever talks to that
protocol: ``opening()`` to start, ``respond()`` for every user message. Adding
the model means writing a second class, not touching the app.

Why parsing lives here rather than in the model
-----------------------------------------------
Whatever produces a value, the value is checked against the contract before it
is stored: :func:`heartrisk.schema.validate_value` runs on every parsed answer,
and a rejected answer is re-asked instead of being kept. That check is what stops
a language model's plausible-sounding invention from reaching the classifier.

Note on units and locale: answers are parsed with Spanish conventions, so
``"5,4"`` is 5.4 and ``"10,000"`` is 10000. A Spanish-formatted thousands
separator written with a dot (``"1.200"``) is read as 1.2; the contract's
plausibility bounds catch the absurd result and the question is re-asked.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from .safety import EMERGENCY_MESSAGE, detect_emergency
from .schema import (
    ASKABLE_FIELDS,
    FIELDS,
    FIELDS_BY_NAME,
    Field,
    SchemaError,
    derive_bmi,
    rejection_message,
    validate_value,
)
from .text import normalise, parse_number, parse_numbers, tokens_of

#: Answers that mean "leave this field out" rather than a value.
SKIP_TOKENS: tuple[str, ...] = (
    "no se",
    "no sé",
    "no lo se",
    "no lo sé",
    "ni idea",
    "no tengo",
    "no la se",
    "no la sé",
    "no me acuerdo",
    "skip",
    "saltear",
    "saltealo",
    "paso",
    "omitir",
    "siguiente",
)

#: Answers that end the interview early.
DONE_TOKENS: tuple[str, ...] = (
    "listo",
    "ya esta",
    "ya está",
    "terminemos",
    "terminar",
    "finalizar",
    "suficiente",
    "es todo",
    "nada mas",
    "nada más",
)

_YES: tuple[str, ...] = ("si", "s", "yes", "claro", "obvio", "correcto", "afirmativo")
_NO: tuple[str, ...] = ("no", "n", "nope", "para nada", "negativo", "jamas")

#: Ordered interpretation rules, checked before the generic yes/no reading and
#: before each other. Order is the semantics: "no fumé nunca" must read as Never,
#: so the negative cues are tested first. Each cue matches a whole word, or a
#: substring only when it is a multi-word phrase. Substring matching on single
#: words is what made "m" match inside "mujer".
_ORDERED_RULES: dict[str, tuple[tuple[tuple[str, ...], str], ...]] = {
    "sex": (
        (("hombre", "masculino", "varon", "m"), "Male"),
        (("mujer", "femenino", "f"), "Female"),
    ),
    "smoker_status": (
        (("nunca", "jamas", "no fumo"), "Never"),
        (("ex", "deje", "fumaba", "antes", "fume"), "Former"),
        (("fumo", "actualmente", "todos los dias", "si"), "Current"),
    ),
    "chest_pain_type": (
        (("nunca", "no siento", "no tengo dolor", "nada", "ninguno"), "Asymptomatic"),
        (("esfuerzo", "cuando hago ejercicio", "al caminar", "caminando"), "Typical Angina"),
        (("reposo", "descanso", "a veces", "irregular"), "Atypical Angina"),
        (("no cardiaco", "puntada", "otra cosa"), "Non-Anginal Pain"),
    ),
}

#: For these fields a bare "no" does not mean ``False``, so the negative reading
#: is explicit instead of implied.
_NEGATIVE_CUES: dict[str, tuple[str, ...]] = {
    "exercise_induced_angina": ("no", "nunca", "para nada", "n"),
    "family_history": ("no", "nadie", "ninguno", "nunca", "n"),
    "wearable_owner": ("no", "nunca", "ninguno", "n"),
}

#: Question order: everything a person can answer from memory first, then the
#: fields that only exist after a measurement. A user who quits early therefore
#: still clears the coverage floor with real answers.
_QUESTION_ORDER: tuple[Field, ...] = tuple(
    f for f in FIELDS if f.name in set(ASKABLE_FIELDS)
) + tuple(f for f in FIELDS if f.name not in set(ASKABLE_FIELDS))


@dataclass(slots=True)
class InterviewState:
    """What the interview knows so far. Lives in ``st.session_state``.

    Attributes:
        answers: contract field name to a validated value.
        skipped: fields the user declined or did not know.
        pending: field currently being asked, if any.
        finished: the interview ended, either completed or cut short.
    """

    answers: dict[str, object] = field(default_factory=dict)
    skipped: set[str] = field(default_factory=set)
    pending: str | None = None
    finished: bool = False

    @property
    def attempted(self) -> set[str]:
        return set(self.answers) | self.skipped

    def next_field(self) -> Field | None:
        """The next field worth asking about, or ``None`` when there is none."""
        for candidate in _QUESTION_ORDER:
            if candidate.name not in self.attempted:
                return candidate
        return None

    def missing_measurements(self) -> tuple[Field, ...]:
        return tuple(
            f for f in FIELDS if not f.askable and f.name not in self.answers
        )


@dataclass(frozen=True, slots=True)
class Turn:
    """One thing the interviewer says back.

    Attributes:
        messages: lines to show the user, in order.
        asking: the field these messages ask about, if any.
        complete: the interview is over.
    """

    messages: tuple[str, ...]
    asking: str | None = None
    complete: bool = False
    #: The interview stopped because the text described an acute event.
    emergency: bool = False


@runtime_checkable
class Interviewer(Protocol):
    """The seam a language-model interviewer implements."""

    def opening(self) -> Turn:
        """The greeting and the first question."""
        ...

    def respond(self, text: str, state: InterviewState) -> Turn:
        """Record an answer and return the next thing to say."""
        ...


# --- parsing ----------------------------------------------------------------


def is_skip(text: str) -> bool:
    """A whole word matches exactly; only a multi-word phrase matches inside.

    Substring matching on single words is how the answer parser once read every
    woman as male (the cue "m" matched inside "mujer"). Skip detection gets the
    same treatment, because a false skip loses a field in silence.
    """
    folded = normalise(text)
    tokens = tokens_of(text)
    return any(
        token in folded if " " in token else token in tokens for token in SKIP_TOKENS
    )


def is_done(text: str) -> bool:
    folded = normalise(text)
    tokens = tokens_of(text)
    return any(
        token in folded if " " in token else token in tokens for token in DONE_TOKENS
    )


def emergency_turn(text: str) -> Turn | None:
    """Return a Turn that stops the interview, when the text describes an acute event.

    Shared by every interviewer implementation, so the safety check cannot be
    forgotten by the language-model one.
    """
    reason = detect_emergency(text)
    if reason is None:
        return None
    return Turn(messages=(EMERGENCY_MESSAGE,), emergency=True)


def _cue_matches(cue: str, folded: str, tokens: set[str]) -> bool:
    """A whole word matches exactly; only a multi-word phrase matches inside."""
    if " " in cue:
        return cue in folded
    return cue in tokens


def parse_boolean(field: Field, text: str) -> bool | None:
    folded = normalise(text)
    tokens = tokens_of(text)

    negative = _NEGATIVE_CUES.get(field.name)
    if negative is not None:
        if any(_cue_matches(cue, folded, tokens) for cue in negative):
            return False
        return True

    if any(_cue_matches(cue, folded, tokens) for cue in _YES):
        return True
    if any(_cue_matches(cue, folded, tokens) for cue in _NO):
        return False
    return None


def parse_category(field: Field, text: str) -> str | None:
    folded = normalise(text)
    tokens = tokens_of(text)

    for category in field.categories:
        if normalise(category) in folded:
            return category

    for cues, category in _ORDERED_RULES.get(field.name, ()):
        if any(_cue_matches(cue, folded, tokens) for cue in cues):
            return category
    return None


def parse_computed(field: Field, text: str) -> float | None:
    """Derive a field the user does not answer directly, such as BMI.

    Every number in one message is read positionally, matching the question's
    own wording ("how much do you weigh and how tall are you").
    """
    numbers = parse_numbers(text)
    if len(numbers) < len(field.computed_from):
        return None
    weight, height = numbers[0], numbers[1]
    return derive_bmi(weight, height)


def parse_answer(field: Field, text: str) -> object:
    """Parse one answer, raising ``SchemaError`` when it cannot be used.

    The message is written to be shown back to the user, so it says what to fix
    rather than naming an internal type.
    """
    if field.computed_from:
        value = parse_computed(field, text)
        if value is None:
            raise SchemaError(
                "Necesito los dos valores: cuánto pesas y cuánto mides."
            )
    elif field.kind == "numeric":
        value = parse_number(text)
        if value is None:
            raise SchemaError(f"Necesito un número para {field.label_es.lower()}.")
    elif field.kind == "boolean":
        value = parse_boolean(field, text)
        if value is None:
            raise SchemaError("Responde con sí o no, por favor.")
    elif field.kind in ("categorical", "binary_category"):
        value = parse_category(field, text)
        if value is None:
            raise SchemaError(
                f"No entendí. Puedes responder con: {', '.join(field.categories)}."
            )
    else:  # pragma: no cover - the contract only declares the kinds above
        raise SchemaError(f"{field.name}: unsupported kind {field.kind!r}")

    try:
        validate_value(field, value)
    except SchemaError as exc:
        # The contract's message is for whoever maintains the pipeline; the person
        # who just typed the value needs the same rule stated in Spanish.
        raise SchemaError(rejection_message(field, value)) from exc
    return value


def question_for(field: Field) -> str:
    """The question, phrased so a lab value never sounds like common knowledge."""
    if field.needs_es:
        return (
            f"{field.question_es} "
            f"Si no lo tienes a mano, dime \"no sé\" y seguimos. "
            f"(Sale de {field.needs_es}.)"
        )
    return f"{field.question_es} Si no lo sabes, dime \"no sé\"."


class StubInterviewer:
    """Deterministic interviewer: ask, parse, validate, repeat.

    Serves two purposes: the app runs with no network or API key, and the tests
    have a stable contract to hold the language-model interviewer against.
    """

    def opening(self) -> Turn:
        return Turn(
            messages=(
                "¡Hola! Soy el asistente de este demo. Voy a hacerte una serie de "
                "preguntas para armar tu perfil.",
                "Puedes responder \"no sé\" en cualquiera y seguimos, o escribir "
                "\"listo\" cuando quieras terminar.",
                question_for(_QUESTION_ORDER[0]),
            ),
            asking=_QUESTION_ORDER[0].name,
        )

    def respond(self, text: str, state: InterviewState) -> Turn:
        emergency = emergency_turn(text)
        if emergency is not None:
            # The pending field is left in place so the interview can resume.
            return emergency

        pending = FIELDS_BY_NAME.get(state.pending) if state.pending else None

        if is_done(text):
            state.pending = None
            state.finished = True
            return Turn(messages=("Listo. Veamos qué tenemos.",), complete=True)

        if pending is not None:
            if is_skip(text):
                state.skipped.add(pending.name)
                state.pending = None
            else:
                try:
                    state.answers[pending.name] = parse_answer(pending, text)
                except SchemaError as exc:
                    # Re-ask the same field rather than storing an unusable value.
                    return Turn(
                        messages=(str(exc), f"De nuevo: {question_for(pending)}"),
                        asking=pending.name,
                    )
                state.pending = None

        following = state.next_field()
        if following is None:
            state.finished = True
            return Turn(
                messages=("Perfecto, con esto ya puedo calcular el resultado.",),
                complete=True,
            )

        state.pending = following.name
        return Turn(messages=(question_for(following),), asking=following.name)
