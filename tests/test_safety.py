"""Tests for the acute-symptom check.

The bias is towards firing, so the cases that must NOT fire matter as much as the
ones that must, and the ordinary interview answers are pinned explicitly: the
safety check and the interview run on the same text.
"""

from __future__ import annotations

import pytest

from heartrisk.interview import InterviewState, StubInterviewer, emergency_turn
from heartrisk.safety import EMERGENCY_MESSAGE, detect_emergency


@pytest.mark.parametrize(
    "text",
    [
        "no puedo respirar",
        "me desmayé hace un rato",
        "siento un dolor que baja al brazo izquierdo",
        "me aprieta el pecho",
        "tengo sudor frío y náuseas",
        "creo que estoy teniendo un infarto",
        "se me nubló la vista",
    ],
)
def test_an_acute_presentation_fires_on_its_own(text: str) -> None:
    assert detect_emergency(text) is not None


@pytest.mark.parametrize(
    "text",
    [
        "me duele el pecho ahora",
        "tengo presión en el pecho desde hace media hora",
        "me falta el aire en este momento",
        "tengo palpitaciones de repente",
    ],
)
def test_an_ambiguous_symptom_fires_with_a_present_tense_marker(text: str) -> None:
    assert detect_emergency(text) is not None


@pytest.mark.parametrize(
    "text",
    [
        "a veces siento dolor en el pecho cuando hago esfuerzo",
        "me falta el aire cuando corro",
        "tengo palpitaciones si tomo mucho café",
        "me duele el pecho",
        "tengo arritmia diagnosticada",
        "tengo taquicardia cuando me pongo nervioso",
    ],
)
def test_an_ambiguous_symptom_alone_does_not_fire(text: str) -> None:
    """These are the answers to the interview's own chest-pain question."""
    assert detect_emergency(text) is None


@pytest.mark.parametrize(
    "text",
    [
        "mi viejo tuvo un infarto a los 55",
        "mi papá tuvo un ataque al corazón",
        "mi madre tuvo problemas del corazón",
        "mi abuelo se murió de un infarto",
    ],
)
def test_describing_a_relative_heart_attack_is_not_an_emergency(text: str) -> None:
    """The family-history question asks for exactly this sentence.

    An earlier version of these rules listed the bare word "infarto" as acute,
    so answering the family-history question stopped the interview and told the
    user to call an ambulance.
    """
    assert detect_emergency(text) is None


@pytest.mark.parametrize(
    "text",
    [
        "sí",
        "no",
        "no sé",
        "nunca fumé",
        "dejé hace dos años",
        "hago 30 minutos por semana",
        "3000 pasos",
        "duermo 6 horas",
        "mi presión es 150/95",
        "hombre",
        "listo",
        "82 kg y 175 cm",
        "",
        "   ",
    ],
)
def test_ordinary_interview_answers_are_not_emergencies(text: str) -> None:
    assert detect_emergency(text) is None


def test_the_message_tells_the_person_what_to_do_and_what_this_is_not() -> None:
    assert "911" in EMERGENCY_MESSAGE
    assert "sala de emergencias" in EMERGENCY_MESSAGE
    assert "no es un servicio médico" in EMERGENCY_MESSAGE


def test_the_emergency_turn_does_not_consume_the_pending_question() -> None:
    """The interview can resume once the person has been told to get help."""
    interviewer = StubInterviewer()
    state = InterviewState(pending="age")

    turn = emergency_turn("me duele el pecho ahora")

    assert turn is not None
    assert turn.emergency
    assert state.pending == "age"
    assert state.answers == {}


def test_the_stub_interviewer_checks_for_emergencies_before_parsing() -> None:
    interviewer = StubInterviewer()
    state = InterviewState(pending="exercise_induced_angina")

    turn = interviewer.respond("sí, me duele el pecho ahora", state)

    assert turn.emergency
    assert state.answers == {}
