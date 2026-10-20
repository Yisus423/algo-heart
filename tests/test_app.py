"""App-level checks.

``AppTest`` executes the real Streamlit script headlessly, so these are the tests
that would catch an app that raises on startup, loses the conversation between
turns, or renders a verdict it is not allowed to give.

They need the frozen artifact, which is not committed because it is a build
output. Run ``python -m heartrisk.train`` once and they run; otherwise they skip
with that instruction.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from heartrisk import llm as llm_module
from heartrisk.artifact import ARTIFACT_PATH
from heartrisk.schema import ASKABLE_FIELDS, FIELDS

APP_PATH = str(Path(__file__).resolve().parents[1] / "app.py")

pytestmark = pytest.mark.skipif(
    not ARTIFACT_PATH.exists(),
    reason="run `python -m heartrisk.train` once; artifacts/ is not committed",
)

#: The order the stub interviewer asks in: everything answerable from memory
#: first, then the fields that only exist after a measurement.
QUESTION_ORDER = [f.name for f in FIELDS if f.name in set(ASKABLE_FIELDS)] + [
    f.name for f in FIELDS if f.name not in set(ASKABLE_FIELDS)
]

#: A complete, plausible answer set. A high-risk profile on purpose, so the
#: happy path exercises a positive verdict rather than the empty case.
ANSWERS = {
    "age": "62",
    "sex": "Male",
    "bmi": "95 kg y 172 cm",
    "chest_pain_type": "cuando hago esfuerzo",
    "exercise_induced_angina": "sí",
    "family_history": "sí",
    "smoker_status": "fumo",
    "alcohol_units_per_week": "8",
    "exercise_minutes_per_week": "30",
    "sleep_hours": "5,5",
    "stress_score": "75",
    "wearable_owner": "sí",
    "daily_steps": "3000",
    "diet_quality_score": "35",
    "resting_bp_systolic": "150",
    "resting_bp_diastolic": "95",
    "cholesterol_total": "240",
    "hdl": "35",
    "ldl": "160",
    "triglycerides": "200",
    "fasting_blood_sugar": "115",
    "hba1c": "6,1",
    "resting_heart_rate": "85",
    "max_heart_rate_achieved": "120",
    "st_depression": "2,0",
}


@pytest.fixture(autouse=True)
def hermetic_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the suite off the network even though a real secrets.toml exists.

    The app reads the deployment's key from `.streamlit/secrets.toml`, which is
    right in production and wrong in tests: a live turn spends quota, needs the
    network, and makes the conversation non-deterministic. Dropping the env var
    was not enough once the credential moved into secrets, which is how these
    tests started calling the real API.

    The environment variable is still honoured, so the one test about a configured
    key keeps exercising the real resolution path.
    """
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    real_resolve = llm_module.resolve_api_key
    monkeypatch.setattr(
        llm_module,
        "resolve_api_key",
        lambda secrets=None, env=None: real_resolve(None, env),
    )


def start() -> AppTest:
    app = AppTest.from_file(APP_PATH, default_timeout=120)
    app.run()
    return app


def say(app: AppTest, text: str) -> None:
    app.chat_input[0].set_value(text).run()


def visible_text(app: AppTest) -> str:
    parts = [element.value for element in app.markdown]
    parts += [element.value for element in app.caption]
    parts += [element.value for element in app.warning]
    parts += [element.value for element in app.info]
    parts += [element.value for element in app.error]
    parts += [element.value for element in app.subheader]
    return " ".join(parts)


def metric(app: AppTest, label: str) -> str:
    for element in app.metric:
        if element.label == label:
            return str(element.value)
    raise AssertionError(f"no metric labelled {label!r}")


# --- the app runs -----------------------------------------------------------


def test_the_app_starts_without_raising() -> None:
    app = start()

    assert not app.exception


def test_the_answer_set_covers_every_contract_field() -> None:
    """Guards the fixture itself: a renamed field must not silently skip a row."""
    assert sorted(ANSWERS) == sorted(QUESTION_ORDER)


def test_the_synthetic_data_disclaimer_is_visible() -> None:
    app = start()
    banner = " ".join(element.value for element in app.error)

    assert "sintéticos" in banner
    assert "No es un dispositivo médico" in banner
    assert "no diagnostica nada" in banner


def test_the_interview_speaks_first() -> None:
    app = start()

    assert "¿Cuántos años tienes?" in visible_text(app)


def test_the_greeting_is_shown_exactly_once() -> None:
    """The opening lines are added by one code path, not two.

    `start_interview` used to pre-fill the history from the turn and then call the
    same helper that appends the turn, so a visitor saw the greeting twice.
    """
    app = start()
    spoken = [element.value for element in app.markdown]

    greetings = [text for text in spoken if text.startswith("¡Hola!")]
    questions = [text for text in spoken if text.startswith("¿Cuántos años")]
    assert len(greetings) == 1
    assert len(questions) == 1


def test_an_answer_advances_to_the_next_question() -> None:
    app = start()

    say(app, "54")

    assert not app.exception
    assert "¿Cuál es tu sexo?" in visible_text(app)


def test_an_unusable_answer_is_asked_again_without_storing_it() -> None:
    app = start()

    say(app, "y… muchos años, che")

    assert not app.exception
    assert "Necesito un número" in visible_text(app)
    assert "¿Cuál es tu sexo?" not in visible_text(app)


# --- the band, which is the point of the demo -------------------------------


def test_stopping_early_shows_the_band_and_refuses_a_verdict() -> None:
    app = start()
    say(app, "54")
    say(app, "Male")
    say(app, "listo")

    assert not app.exception
    assert metric(app, "Campos con datos reales") == "2 / 25"
    assert metric(app, "Confianza") == "sin veredicto"
    text = visible_text(app)
    assert "no alcanza para decirte nada" in text
    assert "Sin veredicto" in text


def test_the_band_names_what_the_user_could_go_and_get_measured() -> None:
    app = start()
    say(app, "54")
    say(app, "listo")

    text = visible_text(app)

    assert "análisis de sangre" in text
    assert "electrocardiograma" in text
    assert "prueba de esfuerzo" in text


def test_fields_filled_in_from_the_training_distribution_are_disclosed() -> None:
    app = start()
    say(app, "54")
    say(app, "listo")

    assert any("Campos completados con promedios" in e.label for e in app.expander)


def test_a_complete_interview_reaches_full_coverage_and_a_verdict() -> None:
    app = start()
    for name in QUESTION_ORDER:
        say(app, ANSWERS[name])

    assert not app.exception
    assert metric(app, "Campos con datos reales") == "25 / 25"
    assert metric(app, "Confianza") == "alta"
    assert "Estimación del modelo" in visible_text(app)


def test_the_verdict_never_claims_a_diagnosis() -> None:
    app = start()
    for name in QUESTION_ORDER:
        say(app, ANSWERS[name])

    verdict = " ".join(element.value for element in app.caption if "Estimación del modelo" in element.value)

    assert "No es un diagnóstico" in verdict
    # No second-person claim that the person has a condition, in any variant.
    for claim in ("puedes tener", "tienes", "tenés", "tú tienes", "padeces"):
        assert claim not in verdict.lower()


def test_the_interview_can_be_restarted() -> None:
    app = start()
    say(app, "54")
    say(app, "listo")

    app.button[0].click().run()

    assert not app.exception
    assert "¿Cuántos años tienes?" in visible_text(app)
    assert not app.metric


# --- emergencies, spec S6 ---------------------------------------------------


def test_an_acute_symptom_stops_the_interview_and_gives_emergency_numbers() -> None:
    app = start()

    say(app, "me duele el pecho ahora")

    assert not app.exception
    banner = " ".join(element.value for element in app.error)
    assert "911" in banner
    assert "sala de emergencias" in banner
    assert "no es un servicio médico" in banner


def test_the_interview_cannot_continue_by_typing_while_an_emergency_is_open() -> None:
    app = start()

    say(app, "me duele el pecho ahora")

    assert not app.chat_input, "the person should not keep typing symptoms into a demo"
    assert any("Seguir con la entrevista" == b.label for b in app.button)


def test_the_interview_resumes_with_the_same_question_after_an_emergency() -> None:
    app = start()
    say(app, "me duele el pecho ahora")

    next(b for b in app.button if b.label == "Seguir con la entrevista").click().run()

    assert not app.exception
    assert app.chat_input, "the interview should be running again"
    resumed = [m.value for m in app.markdown if m.value.startswith("Listo.")]
    assert resumed, "resuming should say something"
    assert "¿Cuántos años tienes?" in resumed[-1], "and re-ask the pending question"


def test_describing_a_relative_who_had_a_heart_attack_is_not_an_emergency() -> None:
    """This is the answer to the family-history question, not a symptom."""
    app = start()
    say(app, "54")
    say(app, "Male")
    say(app, "95 kg y 172 cm")
    say(app, "nunca siento dolor")
    say(app, "no")

    say(app, "mi viejo tuvo un infarto a los 55")

    assert not app.exception
    assert not app.session_state.get("emergency")
    assert "107" not in visible_text(app)


def test_the_engine_is_named_on_screen() -> None:
    app = start()
    say(app, "54")
    say(app, "listo")

    assert "Motor de entrevista: determinista" in visible_text(app)


# --- the credential, spec S10 -----------------------------------------------


def test_without_a_key_the_app_says_so_instead_of_downgrading_silently() -> None:
    app = start()

    warnings = " ".join(element.value for element in app.warning)

    assert "Falta la API key" in warnings
    assert "GEMINI_API_KEY" in warnings
    assert "visitantes no necesitan" in warnings


def test_the_demo_still_runs_when_no_key_is_configured() -> None:
    """A missing credential degrades the engine, it does not break the demo."""
    app = start()

    assert not app.exception
    assert "¿Cuántos años tienes?" in visible_text(app)


def test_a_configured_key_removes_the_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    """No message is sent, so nothing reaches the network; only the wiring is read."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-never-used")

    app = start()

    assert not app.exception
    assert "Falta la API key" not in " ".join(e.value for e in app.warning)
    assert "Gemini" in visible_text(app)
