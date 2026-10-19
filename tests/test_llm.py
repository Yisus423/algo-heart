"""Tests for the language-model interviewer, with no network.

``FakeTransport`` stands in for the network, which is the point of the
``Transport`` seam: the whole understanding path, the validation of what the
model reports, the fallbacks and the emergency short-circuit are all testable
without an API key.
"""

from __future__ import annotations

import pytest

from heartrisk.interview import InterviewState, Turn
from heartrisk.llm import (
    GeminiInterviewer,
    ModelReply,
    build_answer_tool,
    build_interviewer,
    field_schema,
)
from heartrisk.schema import FIELDS, FIELDS_BY_NAME, TARGET


class FakeTransport:
    """Returns scripted replies and records what it was asked to do."""

    def __init__(self, replies: list[ModelReply] | None = None) -> None:
        self.replies = list(replies or [])
        self.calls: list[dict[str, object]] = []

    def generate(self, system, contents, tool):  # noqa: ANN001, ANN201
        self.calls.append({"system": system, "contents": contents, "tool": tool})
        return self.replies.pop(0) if self.replies else ModelReply()


def start(transport: FakeTransport) -> tuple[GeminiInterviewer, InterviewState, Turn]:
    interviewer = GeminiInterviewer(transport)
    state = InterviewState()
    turn = interviewer.opening()
    state.pending = turn.asking
    return interviewer, state, turn


# --- the tool schema is the contract ---------------------------------------


def test_the_tool_covers_every_field_the_interview_can_collect() -> None:
    tool = build_answer_tool()
    properties = tool["parameters_json_schema"]["properties"]  # type: ignore[index]

    expected = {
        f.name for f in FIELDS if not f.computed_from
    }
    assert set(properties) == expected


def test_a_field_the_interview_computes_is_not_something_the_model_reports() -> None:
    """BMI is derived from weight and height, so the model must not guess it."""
    properties = build_answer_tool()["parameters_json_schema"]["properties"]  # type: ignore[index]

    assert "bmi" not in properties
    assert "weight_kg" not in properties
    assert "height_cm" not in properties


def test_a_numeric_field_declares_a_number_and_its_plausible_bounds() -> None:
    schema = field_schema(FIELDS_BY_NAME["age"])

    assert schema["type"] == "number"
    assert schema["minimum"] == 1
    assert schema["maximum"] == 120


def test_a_categorical_field_declares_its_allowed_values() -> None:
    schema = field_schema(FIELDS_BY_NAME["smoker_status"])

    assert schema["type"] == "string"
    assert schema["enum"] == ["Never", "Former", "Current"]


def test_a_boolean_field_declares_a_boolean() -> None:
    assert field_schema(FIELDS_BY_NAME["family_history"])["type"] == "boolean"


def test_a_measurement_field_says_where_the_number_comes_from() -> None:
    assert "análisis de sangre" in field_schema(FIELDS_BY_NAME["hdl"])["description"]


# --- storing what the model understood -------------------------------------


def test_the_model_can_answer_the_field_being_asked_in_natural_language() -> None:
    transport = FakeTransport([ModelReply(text="Anotado.", arguments={"age": 62})])
    interviewer, state, _ = start(transport)

    turn = interviewer.respond("tengo como 62 pirulos, ya soy grande", state)

    assert state.answers["age"] == 62
    assert turn.asking == state.pending
    assert turn.asking != "age"


def test_the_model_can_capture_fields_the_person_volunteered() -> None:
    """This is the whole reason the model is here, not the answer parser."""
    transport = FakeTransport(
        [
            ModelReply(
                text="Buenísimo, gracias por contarme.",
                arguments={
                    "age": 62,
                    "family_history": True,
                    "smoker_status": "Current",
                    "daily_steps": 2500,
                },
            )
        ]
    )
    interviewer, state, _ = start(transport)

    interviewer.respond(
        "Mirá, tengo 62, mi viejo tuvo un infarto, fumo y camino poquísimo", state
    )

    assert state.answers["age"] == 62
    assert state.answers["family_history"] is True
    assert state.answers["smoker_status"] == "Current"
    assert state.answers["daily_steps"] == 2500


def test_an_impossible_value_from_the_model_is_not_stored() -> None:
    transport = FakeTransport([ModelReply(arguments={"age": 500})])
    interviewer, state, _ = start(transport)

    turn = interviewer.respond("tengo 500 años", state)

    assert "age" not in state.answers
    assert turn.asking == "age"


def test_a_rejected_value_is_reported_rather_than_dropped_in_silence() -> None:
    transport = FakeTransport([ModelReply(arguments={"age": 500})])
    interviewer, state, _ = start(transport)

    turn = interviewer.respond("tengo 500 años", state)

    assert any("no los guardé" in message for message in turn.messages)


def test_a_value_the_model_already_knows_is_never_overwritten() -> None:
    transport = FakeTransport([ModelReply(arguments={"age": 99})])
    interviewer, state, _ = start(transport)
    state.answers["age"] = 62

    interviewer.respond("otra vez, tengo 99", state)

    assert state.answers["age"] == 62


def test_the_model_never_reports_a_field_the_contract_does_not_have() -> None:
    transport = FakeTransport([ModelReply(arguments={"pressure": 120, "age": 62})])
    interviewer, state, _ = start(transport)

    interviewer.respond("tengo 62 y presión 120", state)

    assert "pressure" not in state.answers
    assert state.answers["age"] == 62


# --- fallbacks, because a demo must not die on stage ------------------------


def test_the_answer_is_still_captured_when_the_model_says_nothing_useful() -> None:
    """Deterministic parsing backs up the field being asked, not the whole flow."""
    transport = FakeTransport([ModelReply(text="Ok.")])
    interviewer, state, _ = start(transport)

    interviewer.respond("62", state)

    assert state.answers["age"] == 62


def test_the_same_question_is_asked_again_when_nothing_can_be_understood() -> None:
    transport = FakeTransport([ModelReply(text="Disculpá, no te entendí.")])
    interviewer, state, _ = start(transport)

    turn = interviewer.respond("y… ponele que si, qué se yo", state)

    assert "age" not in state.answers
    assert turn.asking == "age"


def test_a_skip_marks_the_field_and_moves_on() -> None:
    transport = FakeTransport([ModelReply(text="Sin problema.")])
    interviewer, state, _ = start(transport)

    turn = interviewer.respond("no sé", state)

    assert "age" in state.skipped
    assert turn.asking != "age"


def test_the_interview_finishes_once_every_field_is_resolved() -> None:
    interviewer, state, _ = start(FakeTransport())
    turn = Turn(messages=(), asking=state.pending)

    for _ in range(60):
        if turn.complete:
            break
        turn = interviewer.respond("no sé", state)

    assert turn.complete
    assert state.finished
    assert len(state.attempted) == len(FIELDS)


def test_the_model_never_sees_the_target_column() -> None:
    """A leaked target would let the interviewer read the answer off the label."""
    tool = build_answer_tool()
    properties = tool["parameters_json_schema"]["properties"]  # type: ignore[index]

    assert TARGET not in properties


# --- safety and wiring ------------------------------------------------------


def test_an_emergency_short_circuits_before_any_model_call() -> None:
    transport = FakeTransport()
    interviewer, state, _ = start(transport)

    turn = interviewer.respond("me duele el pecho ahora", state)

    assert turn.emergency
    assert transport.calls == [], "the model must not be called at all"


def test_the_opening_does_not_need_the_network() -> None:
    transport = FakeTransport()

    turn = GeminiInterviewer(transport).opening()

    assert turn.asking is not None
    assert transport.calls == []


def test_the_model_receives_the_tool_and_the_system_prompt() -> None:
    transport = FakeTransport([ModelReply(text="Ok.")])
    interviewer, state, _ = start(transport)

    interviewer.respond("62", state)

    call = transport.calls[0]
    assert "nunca inventes" in str(call["system"]).lower()
    assert call["tool"]["name"] == "registrar_datos"  # type: ignore[index]


def test_the_conversation_is_sent_back_to_the_model_each_turn() -> None:
    transport = FakeTransport([ModelReply(text="Uno."), ModelReply(text="Dos.")])
    interviewer, state, _ = start(transport)

    interviewer.respond("62", state)
    interviewer.respond("hombre", state)

    assert len(transport.calls[1]["contents"]) >= 2  # type: ignore[arg-type]


def test_without_an_api_key_the_app_falls_back_to_the_deterministic_interviewer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    interviewer = build_interviewer()

    assert type(interviewer).__name__ == "StubInterviewer"


def test_a_key_that_cannot_build_a_client_still_yields_a_working_interviewer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "not-a-real-key")

    interviewer = build_interviewer()

    assert type(interviewer).__name__ == "GeminiInterviewer"
