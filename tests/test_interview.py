"""Tests for the deterministic interviewer.

These pin the contract that a language-model interviewer must also satisfy: the
order of questions, how a skip is handled, that an unusable answer is rejected
and re-asked rather than stored, and when the interview stops.
"""

from __future__ import annotations

import pytest

from heartrisk.interview import (
    StubInterviewer,
    InterviewState,
    is_done,
    is_skip,
    parse_answer,
    parse_category,
    parse_number,
    question_for,
)
from heartrisk.predict import MIN_COVERAGE
from heartrisk.schema import ASKABLE_FIELDS, FIELDS, FIELDS_BY_NAME, SchemaError

ASKABLE_ORDER = [f.name for f in FIELDS if f.name in set(ASKABLE_FIELDS)]


def run_interview(
    answers: dict[str, str] | None = None, *, default: str = "no sé"
) -> tuple[InterviewState, list[str], object]:
    """Drive the whole interview and return the state, the asked order, the turn."""
    interviewer = StubInterviewer()
    state = InterviewState()
    turn = interviewer.opening()
    state.pending = turn.asking
    asked: list[str] = []
    if turn.asking:
        asked.extend(turn.asking)

    guard = 0
    while not turn.complete:
        guard += 1
        assert guard < 200, "interview did not terminate"
        first = turn.asking[0] if turn.asking else ""
        reply = (answers or {}).get(first, default)
        turn = interviewer.respond(reply, state)
        if turn.asking:
            asked.extend(turn.asking)
    return state, asked, turn


# --- opening and order ------------------------------------------------------


def test_the_opening_greets_and_asks_the_first_question() -> None:
    turn = StubInterviewer().opening()

    assert len(turn.messages) > 1
    assert turn.asking == (ASKABLE_ORDER[0],)
    assert not turn.complete


def test_the_core_questions_come_before_the_core_measurements() -> None:
    """The interview never opens by asking for a stress-test value."""
    from heartrisk.schema import CORE_FIELDS, FIELDS_BY_NAME

    _, asked, _ = run_interview()

    core_in_order = [name for name in asked if name in set(CORE_FIELDS)]
    askable = [n for n in core_in_order if FIELDS_BY_NAME[n].askable]
    measured = [n for n in core_in_order if not FIELDS_BY_NAME[n].askable]
    assert core_in_order == askable + measured
    assert asked[0] == "age"


def test_the_interview_asks_only_the_core_and_then_stops() -> None:
    """Chasing all twenty-five was the tedium the user called out."""
    from heartrisk.schema import CORE_FIELDS

    state, asked, turn = run_interview()

    assert turn.complete
    assert state.attempted == set(CORE_FIELDS)
    assert len(asked) == len(CORE_FIELDS)


def test_answering_only_what_a_person_knows_clears_the_coverage_floor() -> None:
    """The interview order is deliberate, not incidental.

    Stopping after the question-answerable fields must land exactly at or above
    the floor where the app is allowed to show a verdict, otherwise a user who
    does not have labs would never see one.
    """
    assert len(ASKABLE_FIELDS) / len(FIELDS) >= MIN_COVERAGE


# --- skips and stops --------------------------------------------------------


def test_a_skip_is_recorded_and_the_field_is_not_asked_again() -> None:
    state, asked, _ = run_interview({"age": "no sé"})

    assert "age" in state.skipped
    assert "age" not in state.answers
    assert asked.count("age") == 1


@pytest.mark.parametrize(
    "text",
    ["no sé", "No se", "ni idea", "skip", "no tengo", "no me acuerdo", "paso"],
)
def test_the_usual_ways_of_not_knowing_count_as_a_skip(text: str) -> None:
    assert is_skip(text)


@pytest.mark.parametrize("text", ["listo", "Ya está", "terminemos", "es todo"])
def test_the_usual_ways_of_stopping_end_the_interview(text: str) -> None:
    interviewer = StubInterviewer()
    state = InterviewState(pending=("age",))

    turn = interviewer.respond(text, state)

    assert is_done(text)
    assert turn.complete
    assert state.finished


def test_stopping_early_keeps_the_answers_already_given() -> None:
    interviewer = StubInterviewer()
    state = InterviewState()
    turn = interviewer.opening()
    state.pending = turn.asking

    interviewer.respond("54", state)
    interviewer.respond("listo", state)

    assert state.answers == {"age": 54}
    assert state.finished


# --- rejecting unusable answers --------------------------------------------


def test_an_answer_without_a_number_is_rejected_and_the_field_is_re_asked() -> None:
    interviewer = StubInterviewer()
    state = InterviewState(pending=("age",))

    turn = interviewer.respond("y... muchos años, che", state)

    assert "age" not in state.answers
    assert turn.asking == ("age",)
    assert not turn.complete


def test_an_impossible_number_is_rejected_and_the_field_is_re_asked() -> None:
    interviewer = StubInterviewer()
    state = InterviewState(pending=("age",))

    turn = interviewer.respond("500", state)

    assert "age" not in state.answers
    assert turn.asking == ("age",)


def test_the_rejection_is_explained_for_the_person_not_for_the_developer() -> None:
    """SchemaError text is English, because whoever maintains the pipeline reads it."""
    interviewer = StubInterviewer()
    state = InterviewState(pending=("age",))

    turn = interviewer.respond("500", state)

    joined = " ".join(turn.messages)
    assert "plausible maximum" not in joined
    assert "fuera del rango razonable" in joined


def test_an_unrecognised_category_is_rejected_and_the_field_is_re_asked() -> None:
    interviewer = StubInterviewer()
    state = InterviewState(pending=("smoker_status",))

    turn = interviewer.respond("depende del día", state)

    assert "smoker_status" not in state.answers
    assert turn.asking == ("smoker_status",)


def test_a_missing_value_is_never_stored() -> None:
    interviewer = StubInterviewer()
    state = InterviewState(pending=("hdl",))

    interviewer.respond("no sé", state)

    assert "hdl" not in state.answers


# --- parsing ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("150", 150.0),
        ("5,4", 5.4),
        ("10,000", 10000.0),
        ("tengo 54 años", 54.0),
        ("82 kg y 175 cm", 82.0),
        ("muchos", None),
    ],
)
def test_numbers_are_read_with_spanish_conventions(text: str, expected: float | None) -> None:
    assert parse_number(text) == expected


def test_a_comma_decimal_is_not_read_as_a_thousands_separator() -> None:
    assert parse_number("mi hba1c es 5,4") == 5.4


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("hombre", "Male"),
        ("Mujer", "Female"),
        ("m", "Male"),
        ("femenino", "Female"),
    ],
)
def test_sex_is_read_from_the_words_people_actually_use(text: str, expected: str) -> None:
    assert parse_category(FIELDS_BY_NAME["sex"], text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("nunca fumé", "Never"),
        ("fumé hasta hace dos años", "Former"),
        ("fumo todos los días", "Current"),
    ],
)
def test_smoking_is_read_from_natural_phrasing(text: str, expected: str) -> None:
    assert parse_category(FIELDS_BY_NAME["smoker_status"], text) == expected


# --- derived values ---------------------------------------------------------


def test_bmi_is_derived_from_the_weight_and_height_the_user_gives() -> None:
    interviewer = StubInterviewer()
    state = InterviewState()
    turn = interviewer.opening()
    state.pending = turn.asking

    interviewer.respond("40", state)
    turn = interviewer.respond("no", state)  # angina de esfuerzo, the second question
    assert turn.asking == ("bmi",), "bmi is the third core question"

    interviewer.respond("82 kg y 175 cm", state)

    assert state.answers["bmi"] == pytest.approx(82 / 1.75**2)


def test_bmi_is_re_asked_when_only_one_of_the_two_values_arrives() -> None:
    interviewer = StubInterviewer()
    state = InterviewState(pending=("bmi",))

    turn = interviewer.respond("82 kilos", state)

    assert "bmi" not in state.answers
    assert turn.asking == ("bmi",)


def test_an_impossible_bmi_is_rejected() -> None:
    with pytest.raises(SchemaError):
        parse_answer(FIELDS_BY_NAME["bmi"], "500 kg y 20 cm")


# --- question wording -------------------------------------------------------


def test_a_lab_value_is_never_asked_as_if_it_were_common_knowledge() -> None:
    question = question_for(FIELDS_BY_NAME["hdl"])

    assert "análisis de sangre" in question
    assert "no sé" in question


def test_a_batch_does_not_repeat_the_hint_after_every_question() -> None:
    """One closing reminder, not three apologies."""
    from heartrisk.interview import batch_question

    joined = batch_question([FIELDS_BY_NAME["age"], FIELDS_BY_NAME["bmi"]])

    assert joined.count("no sé") == 1
    assert joined.endswith('Si algo no lo sabes, dime "no sé" y seguimos.')


def test_a_lifestyle_question_does_not_mention_a_measurement() -> None:
    question = question_for(FIELDS_BY_NAME["daily_steps"])

    assert "Sale de" not in question
    assert "no sé" in question
