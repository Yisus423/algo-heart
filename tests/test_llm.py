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
    DEFAULT_MODEL,
    GeminiTransport,
    call_with_fallback,
    resolve_models,
    SYSTEM_PROMPT,
    GeminiInterviewer,
    ModelReply,
    build_answer_tool,
    build_interviewer,
    field_schema,
    resolve_api_key,
    resolve_model,
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

    expected = {f.name for f in FIELDS if not f.computed_from}
    # Plus the two values the interview collects so the model never divides.
    assert set(properties) == expected | {"weight_kg", "height_cm"}


def test_a_field_the_interview_computes_is_not_something_the_model_reports() -> None:
    """BMI is derived from weight and height, so the model must not guess it."""
    properties = build_answer_tool()["parameters_json_schema"]["properties"]  # type: ignore[index]

    assert "bmi" not in properties, "it is derived, not reported"
    # But the two measurements it is derived from have to be reportable, or the
    # model has no way to pass on "peso 95 y mido 1,72".
    assert "weight_kg" in properties
    assert "height_cm" in properties


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
    assert "age" not in turn.asking


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
    assert "age" in turn.asking


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
    state.pending = ("age",)  # as if the batch had been asked

    interviewer.respond("62", state)

    assert state.answers["age"] == 62


def test_the_same_question_is_asked_again_when_nothing_can_be_understood() -> None:
    transport = FakeTransport([ModelReply(text="Disculpá, no te entendí.")])
    interviewer, state, _ = start(transport)
    state.pending = ("age",)

    turn = interviewer.respond("y… ponele que si, qué se yo", state)

    assert "age" not in state.answers
    assert turn.asking == ("age",)


def test_a_skip_marks_the_field_and_moves_on() -> None:
    transport = FakeTransport([ModelReply(text="Sin problema.")])
    interviewer, state, _ = start(transport)
    state.pending = ("age",)

    turn = interviewer.respond("no sé", state)

    assert "age" in state.skipped
    assert "age" not in turn.asking


def test_the_interview_finishes_once_the_core_is_resolved() -> None:
    """Eight questions, not twenty-five: that was the user's complaint."""
    from heartrisk.schema import CORE_FIELDS

    interviewer, state, turn = start(FakeTransport())

    for _ in range(40):
        if turn.complete:
            break
        state.pending = turn.asking
        turn = interviewer.respond("no sé", state)

    assert turn.complete
    assert state.finished
    assert state.attempted == set(CORE_FIELDS)


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


def test_a_busy_model_switches_to_the_next_one() -> None:
    """A routine 503 on the free tier must not end the interview."""
    tried: list[str] = []

    def attempt(model: str) -> str:
        tried.append(model)
        if model != "second":
            raise RuntimeError("503 UNAVAILABLE: high demand")
        return "answered"

    assert call_with_fallback(attempt, ("first", "second")) == ("answered", "second")
    assert tried == ["first", "second"]


def test_the_model_that_worked_goes_first_next_time() -> None:
    """Retrying the busy one would cost the user a turn for nothing."""
    transport = GeminiTransport(api_key="k", models=("busy", "working"))
    transport.model = "working"

    assert transport.chain() == ("working", "busy")


def test_every_model_failing_raises_so_the_turn_can_degrade() -> None:
    def attempt(model: str) -> str:
        raise RuntimeError("503")

    with pytest.raises(RuntimeError):
        call_with_fallback(attempt, ("a", "b"))


def test_the_configured_model_leads_and_the_measured_ones_follow() -> None:
    """The fallbacks were measured against this key: the 2.5 names answer 404."""
    models = resolve_models({"GEMINI_MODEL": "mine"})

    assert models[0] == "mine"
    assert models[1:] == ("gemini-flash-latest", "gemini-flash-lite-latest")


def test_a_comma_separated_list_of_models_is_accepted() -> None:
    assert resolve_models({"GEMINI_MODELS": "a, b"})[:2] == ("a", "b")


def test_listo_ends_the_interview_on_the_model_path_too() -> None:
    """"listo" only worked in the deterministic path, which made the result screen
    unreachable with a key configured: the model kept re-asking forever."""
    transport = FakeTransport()
    interviewer, state, _ = start(transport)

    turn = interviewer.respond("listo", state)

    assert turn.complete
    assert state.finished
    assert transport.calls == [], "stopping must not need the model"


def test_an_emergency_and_listo_are_shared_by_both_interviewers() -> None:
    """They were duplicated, and the copy in the model path was the one missing."""
    from heartrisk.interview import InterviewState, preflight_turn

    state = InterviewState(pending=("age",))

    assert preflight_turn("listo", state).complete
    assert preflight_turn("me duele el pecho ahora", state).emergency
    assert preflight_turn("tengo 62", state) is None


def test_weight_and_height_from_the_model_become_a_bmi() -> None:
    """BMI is derived in Python, so the model is never asked to divide.

    It was a real bug that it had no way to report them: the tool schema skips `bmi`
    because it is computed, and weight and height are not contract fields, so a
    message containing "peso 95 y mido 1,72" lost both numbers.
    """
    transport = FakeTransport(
        [ModelReply(arguments={"weight_kg": 95, "height_cm": 172})]
    )
    interviewer, state, _ = start(transport)

    interviewer.respond("peso 95 y mido 1,72", state)

    assert state.answers["bmi"] == pytest.approx(95 / 1.72**2)


def test_a_bmi_reported_by_the_model_is_refused() -> None:
    """It is computed from two measurements, so a reported one is ignored."""
    transport = FakeTransport([ModelReply(arguments={"bmi": 22})])
    interviewer, state, _ = start(transport)

    interviewer.respond("creo que tengo 22 de imc", state)

    assert "bmi" not in state.answers


def test_an_impossible_weight_is_not_used_to_derive_a_bmi() -> None:
    transport = FakeTransport(
        [ModelReply(arguments={"weight_kg": 900, "height_cm": 172})]
    )
    interviewer, state, _ = start(transport)

    interviewer.respond("peso 900 kilos", state)

    assert "bmi" not in state.answers


def test_the_extraction_prompt_leaves_the_questions_to_the_system() -> None:
    """Phrasing happens in a second, tool-less call, not in this prompt."""
    assert "Las preguntas las maneja el sistema" in SYSTEM_PROMPT


def test_the_question_is_phrased_in_a_second_call_without_tools() -> None:
    """Spec S12. Gemini returns no text alongside a tool call, so the phrasing
    cannot ride along with the extraction; it needs its own call."""
    transport = FakeTransport(
        [ModelReply(arguments={"age": 62}), ModelReply(text="¿Y tu sexo?")]
    )
    interviewer, state, _ = start(transport)

    turn = interviewer.respond("tengo 62", state)

    assert len(transport.calls) == 2
    assert transport.calls[1]["tool"] is None, "the phrasing call must have no tools"
    assert turn.messages == ("Anotado, gracias.", "¿Y tu sexo?")

    sent = transport.calls[1]["contents"][0]["parts"][0]["text"]  # type: ignore[index]
    assert "sin responderla" in sent, "or the model answers the question itself"
    assert "¿Sientes dolor o presión en el pecho cuando haces esfuerzo físico?" in sent, "and it is told which question to rephrase"


def test_the_contracts_wording_is_used_when_the_phrasing_call_returns_nothing() -> None:
    """The person must never be left without a question to answer."""
    transport = FakeTransport([ModelReply(arguments={"age": 62}), ModelReply()])
    interviewer, state, _ = start(transport)

    turn = interviewer.respond("tengo 62", state)

    assert turn.messages[-1].startswith("¿Sientes dolor o presión en el pecho cuando haces esfuerzo físico?")


def test_a_reply_without_a_question_mark_is_not_used_as_the_question() -> None:
    transport = FakeTransport(
        [ModelReply(arguments={"age": 62}), ModelReply(text="ok")]
    )
    interviewer, state, _ = start(transport)

    turn = interviewer.respond("tengo 62", state)

    assert turn.messages[-1].startswith("¿Sientes dolor o presión en el pecho cuando haces esfuerzo físico?")


def test_a_failing_model_degrades_instead_of_crashing() -> None:
    """A 503 in front of an audience must not be a traceback on screen."""

    class Exploding:
        def generate(self, system, contents, tool):  # noqa: ANN001, ANN201
            raise RuntimeError("503 UNAVAILABLE: high demand")

    interviewer = GeminiInterviewer(Exploding())
    state = InterviewState()
    turn = interviewer.opening()
    state.pending = turn.asking

    state.pending = ("age",)  # as if the batch had been asked
    turn = interviewer.respond("62", state)

    assert state.answers["age"] == 62, "the answer must not be lost"
    assert "cortó la conexión" in turn.messages[0]
    assert turn.asking, "and the interview keeps moving"


def test_the_app_acknowledges_when_the_model_returns_no_text() -> None:
    transport = FakeTransport([ModelReply(arguments={"age": 62})])
    interviewer, state, _ = start(transport)

    turn = interviewer.respond("tengo 62", state)

    assert any("Anotado" in message for message in turn.messages)


def test_nothing_is_acknowledged_when_no_value_was_registered() -> None:
    transport = FakeTransport([ModelReply()])
    interviewer, state, _ = start(transport)

    turn = interviewer.respond("no sé", state)

    assert not any("Anotado" in message for message in turn.messages)


def test_a_rejected_value_is_explained_in_spanish_not_in_developer_english() -> None:
    """A SchemaError is written for whoever maintains the pipeline, not for the user."""
    transport = FakeTransport([ModelReply(arguments={"age": 500})])
    interviewer, state, _ = start(transport)

    turn = interviewer.respond("tengo 500 años", state)

    report = " ".join(turn.messages)
    assert "plausible maximum" not in report
    assert "fuera del rango razonable" in report


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

    extraction = [call for call in transport.calls if call["tool"] is not None]
    assert len(extraction[1]["contents"]) >= 2, "the transcript is resent each turn"


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


# --- where the credential comes from ---------------------------------------


def test_secrets_win_over_the_environment() -> None:
    """The deployment's configuration must not be overridden by a local export."""
    resolved = resolve_api_key(
        {"GEMINI_API_KEY": "from-secrets"}, {"GEMINI_API_KEY": "from-env"}
    )

    assert resolved == "from-secrets"


def test_the_environment_is_used_when_secrets_declare_nothing() -> None:
    assert resolve_api_key({}, {"GEMINI_API_KEY": "from-env"}) == "from-env"


def test_both_spellings_of_the_key_are_accepted() -> None:
    assert resolve_api_key({"GEMINI_API_KEY": "upper"}) == "upper"
    assert resolve_api_key({"gemini_api_key": "lower"}) == "lower"


def test_an_absent_or_blank_key_resolves_to_none() -> None:
    assert resolve_api_key({}, {}) is None
    assert resolve_api_key({"GEMINI_API_KEY": "   "}, {}) is None
    assert resolve_api_key(None, {}) is None


def test_the_model_name_can_come_from_secrets() -> None:
    assert resolve_model({"GEMINI_MODEL": "some-model"}) == "some-model"


def test_the_model_falls_back_to_the_default() -> None:
    assert resolve_model({}, {}) == DEFAULT_MODEL


def test_a_visitor_brings_no_credential_of_their_own() -> None:
    """The seam that matters: the deployment supplies the key, the app reads it."""
    interviewer = build_interviewer(secrets={"GEMINI_API_KEY": "deployment-key"})

    assert type(interviewer).__name__ == "GeminiInterviewer"
