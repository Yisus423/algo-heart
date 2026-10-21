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
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from .safety import EMERGENCY_MESSAGE, detect_emergency
from .schema import (
    ASKABLE_FIELDS,
    CORE_FIELDS,
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
#: Core questions a person answers from memory, in measured order but with age
#: first. Age is the one thing everybody can answer instantly and it sets the
#: context for the rest; an interview whose first question is "do you have the max
#: heart rate from a stress test" is not an interview.
_CORE_ASKABLE_ORDER: tuple[str, ...] = ("age",) + tuple(
    name
    for name in CORE_FIELDS
    if FIELDS_BY_NAME[name].askable and name != "age"
)

#: Everything the interview asks, in the order it asks it: the core questions from
#: memory, then the core measurements, then the fields that are only accepted if
#: the person volunteers them.
_QUESTION_ORDER: tuple[Field, ...] = (
    tuple(FIELDS_BY_NAME[name] for name in _CORE_ASKABLE_ORDER)
    + tuple(
        FIELDS_BY_NAME[name] for name in CORE_FIELDS if not FIELDS_BY_NAME[name].askable
    )
    + tuple(f for f in FIELDS if f.name not in set(CORE_FIELDS))
)

#: How many questions the interview asks in one turn when it has a model that can
#: match several answers to several questions at once. See `next_batch`.
BATCH_SIZE = 3


def question_order() -> tuple[Field, ...]:
    """Every field in the order the interview asks them: answerable ones first.

    Public because the language-model interviewer needs to predict which question
    comes next before it calls the model, so it can hand that question over to be
    phrased rather than pasted.
    """
    return _QUESTION_ORDER


@dataclass(slots=True)
class InterviewState:
    """What the interview knows so far. Lives in ``st.session_state``.

    Attributes:
        answers: contract field name to a validated value.
        skipped: fields the user declined or did not know.
        pending: fields asked in the last turn and not yet resolved.
        finished: the interview ended, either completed or cut short.
    """

    answers: dict[str, object] = field(default_factory=dict)
    skipped: set[str] = field(default_factory=set)
    pending: tuple[str, ...] = ()
    finished: bool = False

    @property
    def attempted(self) -> set[str]:
        return set(self.answers) | self.skipped

    def next_field(self) -> Field | None:
        """The next core field to ask about, or ``None`` when there is none.

        Only the core is chased. Anything else is welcome if the person volunteers
        it, but the interview does not spend a turn on a field worth 0.35% of the
        model.
        """
        for candidate in _QUESTION_ORDER:
            if candidate.name in set(CORE_FIELDS) and candidate.name not in self.attempted:
                return candidate
        return None

    def next_batch(self) -> tuple[Field, ...]:
        """The next core fields as a single ask, grouped by where the numbers come from.

        Grouping is what makes a batch read as one question instead of three
        demands: the numbers that come out of one blood test belong together,
        systolic pressure does not belong with minutes of exercise.
        """
        missing = [
            f.name
            for f in _QUESTION_ORDER
            if f.name in set(CORE_FIELDS) and f.name not in self.attempted
        ]
        if not missing:
            return ()
        head = FIELDS_BY_NAME[missing[0]]
        together = [n for n in missing if _same_source(FIELDS_BY_NAME[n], head)]
        return tuple(FIELDS_BY_NAME[n] for n in together[:BATCH_SIZE])

    def missing_measurements(self) -> tuple[Field, ...]:
        """Core fields that need a measurement and are still unknown."""
        return tuple(
            FIELDS_BY_NAME[name]
            for name in CORE_FIELDS
            if not FIELDS_BY_NAME[name].askable and name not in self.answers
        )


def _same_source(a: Field, b: Field) -> bool:
    """Whether two fields come from the same place, so one ask covers both."""
    if a.askable and b.askable:
        return True
    if not a.askable and not b.askable:
        return a.source_es == b.source_es
    return False


def batch_question(fields: Sequence[Field]) -> str:
    """The contract's own wording for a whole batch, for paths that cannot phrase one.

    One closing reminder instead of one per question.
    """
    body = " ".join(question_for(f, hint=False) for f in fields)
    return f"{body} Si algo no lo sabes, dime \"no sé\" y seguimos."


@dataclass(frozen=True, slots=True)
class Turn:
    """One thing the interviewer says back.

    Attributes:
        messages: lines to show the user, in order.
        asking: the fields these messages ask about, if any.
        complete: the interview is over.
    """

    messages: tuple[str, ...]
    #: Every field this turn asks about. More than one when the turn asks a batch.
    asking: tuple[str, ...] = ()
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


def preflight_turn(text: str, state: InterviewState) -> Turn | None:
    """The two checks that run before anything else, for every interviewer.

    They used to live only in the deterministic path. The deterministic interviewer
    honoured "listo" and the language-model one did not, so with a key configured the
    interview could never be cut short and the result screen was unreachable — a
    divergence between two implementations of the same interface, which is exactly
    what a shared helper prevents.
    """
    emergency = emergency_turn(text)
    if emergency is not None:
        return emergency
    if is_done(text):
        state.pending = ()
        state.finished = True
        return Turn(messages=("Listo. Veamos qué tenemos.",), complete=True)
    return None


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


def question_for(field: Field, *, hint: bool = True) -> str:
    """The question, phrased so a lab value never sounds like common knowledge.

    ``hint=False`` drops the "say so if you do not know" reminder. A batch carries
    that reminder once at the end; after every question it turns one ask into three
    apologies.
    """
    source = f" (Sale de {field.needs_es}.)" if field.needs_es else ""
    if not hint:
        return f"{field.question_es}{source}"
    if field.needs_es:
        return (
            f"{field.question_es} "
            f"Si no lo tienes a mano, dime \"no sé\" y seguimos. "
            f"(Sale de {field.needs_es}.)"
        )
    return f"{field.question_es} Si no lo sabes, dime \"no sé\"."


class StubInterviewer:
    """Deterministic interviewer: ask, parse, validate, repeat.

    It asks one field per turn and does not use the free-form opening. Matching
    several answers in one message to several questions takes understanding, and
    guessing wrong files a number under the wrong question, which is worse than
    asking again. The language-model interviewer is the one that batches.
    """

    def opening(self) -> Turn:
        first = _QUESTION_ORDER[0]
        return Turn(
            messages=(
                "¡Hola! Soy el asistente de este demo. Te haré unas pocas preguntas.",
                "Puedes responder \"no sé\" en cualquiera y seguimos, o escribir "
                "\"listo\" cuando quieras terminar.",
                question_for(first),
            ),
            asking=(first.name,),
        )

    def respond(self, text: str, state: InterviewState) -> Turn:
        stop = preflight_turn(text, state)
        if stop is not None:
            return stop

        pending = tuple(FIELDS_BY_NAME[name] for name in state.pending)

        if pending:
            asked = pending[0]
            if is_skip(text):
                state.skipped.add(asked.name)
            else:
                try:
                    state.answers[asked.name] = parse_answer(asked, text)
                except SchemaError as exc:
                    # Re-ask the same field rather than storing an unusable value.
                    return Turn(
                        messages=(str(exc), f"De nuevo: {question_for(asked)}"),
                        asking=(asked.name,),
                    )
            state.pending = ()

        following = state.next_field()
        if following is None:
            state.finished = True
            return Turn(
                messages=("Perfecto, con esto ya puedo calcular el resultado.",),
                complete=True,
            )

        state.pending = (following.name,)
        return Turn(messages=(question_for(following),), asking=(following.name,))
