"""Streamlit app: interview a person, then show how solid the answer actually is.

Run it with ``streamlit run app.py`` after building the artifact with
``python -m heartrisk.train``.

The presentation order is the point of this file. The count of real fields and
the confidence band come FIRST and occupy the width; the model's verdict comes
last, small, and worded so it cannot be read as a diagnosis. When the interview
did not collect enough to say anything, no verdict is shown at all and the screen
lists what the person could go and get measured instead.

Colours come from ``.streamlit/config.toml``, sampled from screenshots of the
Algorithmics course page. The typeface needs one style block because Streamlit's
``[theme] fontFaces`` requires pinned font-file URLs; see ``FONT_STYLE``.
"""

from __future__ import annotations

from collections import defaultdict

import streamlit as st

from heartrisk.artifact import Artifact, load_artifact
from heartrisk.interview import InterviewState, Interviewer, Turn, question_for
from heartrisk.llm import API_KEY_ENV, GeminiInterviewer, build_interviewer
from heartrisk.predict import Confidence, Prediction, predict
from heartrisk.safety import EMERGENCY_MESSAGE
from heartrisk.schema import FIELDS_BY_NAME

# Design tokens sampled from the course page that [theme] cannot express.
DEEP_PURPLE = "#602B7A"
INDIGO = "#220E56"
SURFACE_ALT = "#F7F9FA"
CHIP_NEUTRAL = "#E8ECF0"
CREAM = "#FFF4E2"
MINT = "#E6FFE2"

#: Spec S6 and S7. Persistent, never dismissible, never behind an expander.
DISCLAIMER = (
    "**Demo académico.** Este modelo fue entrenado con datos **sintéticos** "
    "(pacientes inventados por un generador, no personas reales). **No es un "
    "dispositivo médico y no diagnostica nada.** Si tenés síntomas o dudas sobre "
    "tu salud, consultá a un médico."
)

#: The heading family is an INFERENCE from the screenshots, not a measured value.
#: Streamlit's [theme] fontFaces wants pinned font-file URLs, so the family is
#: pulled from Google Fonts here instead: the single place theming cannot express
#: what we need.
FONT_STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800&display=swap');
html, body, [class*="st-"] { font-family: 'Nunito', system-ui, sans-serif; }
h1, h2, h3, h4 { font-family: 'Nunito', system-ui, sans-serif; font-weight: 800; }
</style>
"""

CONFIDENCE_LABEL = {
    Confidence.NONE: "sin veredicto",
    Confidence.LOW: "baja",
    Confidence.MEDIUM: "media",
    Confidence.HIGH: "alta",
}

CONFIDENCE_NOTE = {
    Confidence.LOW: (
        "Casi todo lo que sabemos viene de tus respuestas. Los valores que faltan "
        "se completaron con promedios del dataset, así que tomalo como una "
        "aproximación gruesa."
    ),
    Confidence.MEDIUM: (
        "Buena parte del perfil tiene valores reales, pero todavía hay campos "
        "completados con promedios."
    ),
    Confidence.HIGH: (
        "Casi todos los campos tienen valores reales. Dentro de lo que este dataset "
        "sintético permite, es lo más sólido que este demo puede dar."
    ),
}


#: Shown in the conversation where it happened, while the full message goes to
#: the banner at the top of the screen.
EMERGENCY_CHAT_LINE = (
    "**Pará un segundo.** Te dejo las indicaciones arriba, en rojo. "
    "Lo que me contaste no es algo que este demo pueda evaluar."
)


@st.cache_resource(show_spinner=False)
def cached_artifact() -> Artifact:
    """Load the frozen artifact once per session."""
    return load_artifact()


def load_model() -> Artifact | None:
    """Return the artifact, or explain how to build it instead of crashing."""
    try:
        return cached_artifact()
    except FileNotFoundError:
        st.error(
            "Todavía no hay un modelo entrenado. Corré esto una vez desde la raíz "
            "del proyecto:\n\n"
            "```bash\npython -m heartrisk.train\n```"
        )
        return None


def interviewer() -> Interviewer:
    """This session's interviewer.

    It lives in ``session_state`` because the language-model implementation keeps
    the conversation transcript on the instance; recreating it on every rerun
    would make it forget what the user already said.
    """
    return st.session_state.interviewer


def start_interview() -> None:
    """Reset the conversation and speak first."""
    state = InterviewState()
    st.session_state.interviewer = build_interviewer()
    turn = interviewer().opening()
    st.session_state.messages = [("assistant", line) for line in turn.messages]
    st.session_state.interview = state
    st.session_state.result = None
    st.session_state.emergency = False
    apply_turn(turn)


def apply_turn(turn: Turn) -> None:
    """Record what the interviewer just said.

    ``respond`` already updates ``state.pending``; ``opening`` cannot, because it
    takes no state. Re-stating it here keeps both paths identical and makes the
    caller the single place that moves the conversation forward.
    """
    if turn.emergency:
        # The pending question is deliberately left pending, so the interview can
        # resume once the person has been told to get help.
        st.session_state.messages.append(("assistant", EMERGENCY_CHAT_LINE))
        st.session_state.emergency = True
        return

    st.session_state.messages.extend(("assistant", line) for line in turn.messages)
    if turn.asking is not None:
        st.session_state.interview.pending = turn.asking
    if turn.complete:
        st.session_state.result = compute_result()


def compute_result() -> Prediction | None:
    artifact = load_model()
    if artifact is None:
        return None
    payload = dict(st.session_state.interview.answers)
    if not payload:
        return None
    return predict(payload, artifact)


def render_band(result: Prediction) -> None:
    """Show coverage and confidence first, big; the verdict last, small."""
    st.subheader("Qué tan sólido es este resultado")

    coverage_column, confidence_column = st.columns([2, 1])
    with coverage_column:
        st.metric(
            "Campos con datos reales",
            f"{result.provided_count} / {result.total_count}",
            help="Cada campo sin dato real se completa con un promedio del dataset.",
        )
        st.progress(result.coverage)
    with confidence_column:
        st.metric("Confianza", CONFIDENCE_LABEL[result.confidence])

    if not result.estimate_available:
        st.warning(
            "Con esta cantidad de datos **no alcanza para decirte nada**. "
            "No hay veredicto: la respuesta honesta es que falta información."
        )
    else:
        st.info(CONFIDENCE_NOTE[result.confidence])

    if result.missing_measurements:
        st.markdown("#### Lo que falta, y de dónde sale")
        grouped: dict[str, list[str]] = defaultdict(list)
        for name in result.missing_measurements:
            contract_field = FIELDS_BY_NAME[name]
            grouped[contract_field.needs_es or "otra medición"].append(
                contract_field.label_es
            )
        for need, labels in grouped.items():
            st.markdown(f"- **{need}** — {', '.join(sorted(labels))}")
        st.caption(
            "Estos campos no son cosas que uno recuerde de memoria: salen de un "
            "estudio. El modelo los completa con promedios, así que la estimación "
            "mejora mucho si los tenés."
        )

    if result.out_of_distribution:
        unusual = ", ".join(
            FIELDS_BY_NAME[name].label_es for name in result.out_of_distribution
        )
        st.warning(
            f"Estos valores están fuera del rango que el modelo vio al entrenar: "
            f"{unusual}. La estimación es menos confiable para este perfil."
        )

    with st.expander(f"Campos completados con promedios ({len(result.imputed)})"):
        if not result.imputed:
            st.write("Ninguno: todos los campos tienen un valor real.")
        else:
            for name in result.imputed:
                st.markdown(
                    f"- {FIELDS_BY_NAME[name].label_es}"
                    f"{f' ({FIELDS_BY_NAME[name].unit_es})' if FIELDS_BY_NAME[name].unit_es else ''}"
                )

    st.divider()
    render_verdict(result)


def render_verdict(result: Prediction) -> None:
    """The verdict, deliberately quieter than the band above it."""
    if result.label is None:
        st.caption(
            "**Sin veredicto.** Con los datos que juntamos, cualquier número que te "
            "diera sería inventado."
        )
        return

    if result.label == 1:
        st.caption(
            "**Estimación del modelo: perfil del lado de mayor riesgo** en los datos "
            "sintéticos con los que fue entrenado. No es un diagnóstico, no dice que "
            "tengas ninguna condición, y no reemplaza una consulta médica."
        )
    else:
        st.caption(
            "**Estimación del modelo: perfil del lado de menor riesgo.** No es un "
            "diagnóstico, y un resultado así **no descarta nada**: este modelo deja "
            "pasar cerca de 1 de cada 7 casos en sus propios datos de prueba."
        )


def engine_label(active: Interviewer) -> str:
    if isinstance(active, GeminiInterviewer):
        return "Gemini (entiende respuestas en lenguaje natural)"
    return f"determinista (sin {API_KEY_ENV})"


def main() -> None:
    st.set_page_config(
        page_title="Riesgo cardíaco — demo académico",
        page_icon="🫀",
        layout="centered",
    )
    st.markdown(FONT_STYLE, unsafe_allow_html=True)
    st.error(DISCLAIMER, icon="⚠️")

    st.title("Riesgo cardíaco")
    st.caption(
        "Contame tus hábitos y, si los tenés, tus números. Al final te muestro qué "
        "tan sólido es el resultado y qué te convendría medir."
    )

    if load_model() is None:
        return

    if "messages" not in st.session_state:
        start_interview()

    if isinstance(interviewer(), GeminiInterviewer):
        st.info(
            "Para interpretar tus respuestas, este demo manda el texto a Google "
            "(plan gratuito), que puede usarlo para mejorar sus modelos. Es un "
            "demo: no escribas datos reales de salud.",
            icon="🔒",
        )
    st.caption(f"Motor de entrevista: {engine_label(interviewer())}")

    if st.session_state.get("emergency"):
        st.error(EMERGENCY_MESSAGE, icon="🚑")

    result: Prediction | None = st.session_state.result
    if result is not None:
        render_band(result)

    st.divider()

    for role, text in st.session_state.messages:
        with st.chat_message(role):
            st.markdown(text)

    if st.session_state.get("emergency"):
        if st.button("Seguir con la entrevista", type="primary"):
            st.session_state.emergency = False
            pending = st.session_state.interview.pending
            follow_up = (
                question_for(FIELDS_BY_NAME[pending]) if pending else "Seguimos."
            )
            st.session_state.messages.append(("assistant", f"Listo. {follow_up}"))
            st.rerun()
        return

    if st.session_state.interview.finished:
        st.success("La entrevista terminó.", icon="✅")
        if st.button("Empezar de nuevo", type="primary"):
            start_interview()
            st.rerun()
        return

    prompt = st.chat_input("Escribí tu respuesta…")
    if prompt:
        st.session_state.messages.append(("user", prompt))
        apply_turn(interviewer().respond(prompt, st.session_state.interview))
        st.rerun()


# Streamlit executes this file as a script, so the app starts unconditionally.
main()
