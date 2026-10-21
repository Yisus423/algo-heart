"""Streamlit app: interview a person, then show how solid the answer actually is.

Run it with ``streamlit run app.py`` after building the artifact with
``python -m heartrisk.train``.

The presentation order is the point of this file. The count of real fields and
the confidence band come FIRST and occupy the width; the model's verdict comes
last, small, and worded so it cannot be read as a diagnosis. When the interview
did not collect enough to say anything, no verdict is shown at all and the screen
lists what the person could go and get measured instead.

Colours and the typeface come from ``.streamlit/config.toml``, sampled from
screenshots of the Algorithmics course page. Nothing is injected as raw HTML: a
blanket CSS override broad enough to set the page font also replaces the icon font
Streamlit ships locally, which is what once broke every Material icon in the
interface. See ``tests/test_ui.py``.
"""

from __future__ import annotations

from collections import defaultdict

import streamlit as st

from heartrisk.artifact import Artifact, load_artifact
from heartrisk.interview import InterviewState, Interviewer, Turn, question_for
from heartrisk.llm import API_KEY_ENV, GeminiInterviewer, build_interviewer
from heartrisk.predict import Confidence, Prediction, predict
from heartrisk.safety import EMERGENCY_MESSAGE
from heartrisk.schema import CORE_FIELDS, FIELDS_BY_NAME, source_group

# Design tokens sampled from the course page that [theme] cannot express.
DEEP_PURPLE = "#602B7A"
INDIGO = "#220E56"
SURFACE_ALT = "#F7F9FA"
CHIP_NEUTRAL = "#E8ECF0"
CREAM = "#FFF4E2"
MINT = "#E6FFE2"

#: The alarm stays, the paragraph does not: one red line, and the detail in a
#: caption underneath. S6 needs the disclaimer read, not the whole first screen
#: taken. See L40 in the feature document.
DISCLAIMER = "**Demo académico con datos sintéticos. No diagnostica nada.**"
DISCLAIMER_DETAIL = (
    "No es un dispositivo médico. Si tienes síntomas o dudas sobre tu salud, "
    "consulta a un médico."
)

#: Spanish for the values the contract stores in English, for the checklist.
CATEGORY_ES: dict[str, dict[str, str]] = {
    "sex": {"Female": "mujer", "Male": "hombre"},
    "smoker_status": {"Never": "nunca fumó", "Former": "ex fumador", "Current": "fuma"},
    "chest_pain_type": {
        "Asymptomatic": "sin dolor",
        "Atypical Angina": "angina atípica",
        "Non-Anginal Pain": "dolor no cardíaco",
        "Typical Angina": "angina típica",
    },
}

#: Group heading for the fields a person answers from memory.
FROM_MEMORY = "De tus respuestas"

VERDICT_TITLE = {
    1: "Perfil del lado de mayor riesgo",
    0: "Perfil del lado de menor riesgo",
}

VERDICT_NOTE = {
    1: "Estimación de un modelo entrenado con datos **sintéticos**. No es un "
    "diagnóstico, no dice que tengas ninguna condición, y no reemplaza una consulta "
    "médica.",
    0: "Estimación de un modelo entrenado con datos **sintéticos**. No es un "
    "diagnóstico, y un resultado así **no descarta nada**: este modelo deja pasar "
    "cerca de 1 de cada 7 casos en sus propios datos de prueba.",
}

CONFIDENCE_LABEL = {
    Confidence.NONE: "sin veredicto",
    Confidence.LOW: "baja",
    Confidence.MEDIUM: "media",
    Confidence.HIGH: "alta",
}

CONFIDENCE_NOTE = {
    Confidence.LOW: (
        "Casi todo lo que sabemos viene de tus respuestas. Los valores que faltan "
        "se completaron con promedios del dataset, así que tómalo como una "
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
    "**Detente un segundo.** Las indicaciones están arriba, en rojo. "
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
            "Todavía no hay un modelo entrenado. Ejecuta esto una vez desde la raíz "
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


def secret_map() -> dict[str, object]:
    """Read Streamlit secrets as a plain mapping.

    Streamlit raises ``StreamlitSecretNotFoundError``, which subclasses
    ``FileNotFoundError``, when the deployment has no secrets file. That is a
    configuration state rather than a crash, and ``main`` decides how loudly to
    report it. Any other error is left to surface: this app does not want to
    swallow a real configuration fault.
    """
    try:
        return dict(st.secrets.items())
    except FileNotFoundError:
        return {}


def start_interview() -> None:
    """Reset the conversation and speak first.

    The interviewer is built from the deployment's secrets, so a visitor brings no
    credential of their own.

    The opening lines are added only by ``apply_turn``. An earlier version also
    pre-filled ``messages`` from the same turn, so the greeting was appended twice
    and a visitor read it twice.
    """
    state = InterviewState()
    st.session_state.interviewer = build_interviewer(secrets=secret_map())
    st.session_state.interview = state
    st.session_state.messages = []
    st.session_state.result = None
    st.session_state.emergency = False
    apply_turn(interviewer().opening())


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

    spoken = "\n\n".join(line for line in turn.messages if line)
    if spoken:
        # One bubble per turn, not one per line. Three separate bubbles with three
        # avatars for a single opening read as three different speakers, and every
        # model turn produced two of them, so the history stopped looking like a
        # conversation at all.
        st.session_state.messages.append(("assistant", spoken))
    if turn.asking:
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
    """The result, as the protagonist of the screen.

    The user, looking at the previous layout: "se ve que hacemos más énfasis en los
    datos recolectados que en el resultado (casi ni se encuentra el resultado)". It
    was true: the verdict was a grey caption under an expander and a divider, while
    the loudest element on screen was a bold checklist of answers the person had
    already given. Here the verdict is a heading with its own name, and coverage is
    supporting detail rather than the headline.

    The progress bar is gone on purpose. Inside a bordered container it measured
    itself against the wrong width and rendered 8/8 as two thirds full; the metric
    already states the number without lying.
    """
    with st.container(border=True):
        st.caption("TU RESULTADO")

        if result.label is None:
            st.subheader("Sin veredicto")
            st.markdown(
                "Con esta cantidad de datos **no alcanza para decirte nada**. Un "
                "número aquí sería inventado."
            )
        else:
            st.subheader(VERDICT_TITLE[result.label])
            st.markdown(VERDICT_NOTE[result.label])

        st.divider()
        confidence_column, coverage_column = st.columns(2)
        with confidence_column:
            st.metric("Confianza", CONFIDENCE_LABEL[result.confidence])
        with coverage_column:
            st.metric(
                "Datos reales de los que pedimos",
                f"{result.provided_count} / {result.total_count}",
                help="Cada campo sin dato real se completa con un promedio del dataset.",
            )
        if result.beyond_core:
            st.caption(
                f"Y {len(result.beyond_core)} dato(s) más que aportaste y no pedimos: "
                "cuentan para el resultado igual."
            )

        if result.estimate_available:
            st.info(CONFIDENCE_NOTE[result.confidence])

        if result.out_of_distribution:
            unusual = ", ".join(
                FIELDS_BY_NAME[name].label_es for name in result.out_of_distribution
            )
            st.warning(
                f"Estos valores están fuera del rango que el modelo vio al entrenar: "
                f"{unusual}. La estimación es menos confiable para este perfil."
            )

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

        with st.expander(f"Campos completados con promedios ({len(result.imputed)})"):
            if not result.imputed:
                st.write("Ninguno: todos los campos tienen un valor real.")
            else:
                for name in result.imputed:
                    contract_field = FIELDS_BY_NAME[name]
                    unit = f" ({contract_field.unit_es})" if contract_field.unit_es else ""
                    st.markdown(f"- {contract_field.label_es}{unit}")


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


def format_value(contract_field, value: object) -> str:
    """The captured value as a person would write it, not as the contract stores it.

    The contract keeps categoricals in English (``Male``, ``Never``) because those
    are the dataset's own labels, and showing them raw would put dataset vocabulary
    in front of the user.
    """
    if contract_field.kind == "boolean":
        return "sí" if value else "no"
    if contract_field.kind in ("categorical", "binary_category"):
        return CATEGORY_ES.get(contract_field.name, {}).get(str(value), str(value))
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        # One decimal for a measurement, none for a whole number: "32.1 kg/m²" and
        # "62 años", never "32.11243333333333 kg/m²" in front of an audience.
        number = f"{value:g}" if float(value).is_integer() else f"{value:.1f}"
        return f"{number} {contract_field.unit_es}" if contract_field.unit_es else number
    return str(value)


def render_data_checklist(state: InterviewState) -> None:
    """Spec S14: what is already known, and where each missing number comes from.

    Grouped by origin rather than by contract order, so the panel doubles as the
    answer to "what should I go and get measured" and not only as a progress
    readout. This is the part of the demo with teaching value: the strongest
    predictors in this dataset cannot be guessed, they have to be measured.

    It lives in its own column, beside the conversation. In the main column it took
    most of the screen and pushed the chat below the fold; in a narrow centred
    layout the rest of the page was empty. A column does both jobs.
    """
    # Only the core is listed. Twenty-five rows turned the checklist into a wall, and
    # seventeen of those fields are not asked for anyway: they are accepted if the
    # person brings them, which is what the caption at the end says.
    core = [FIELDS_BY_NAME[name] for name in CORE_FIELDS]
    groups: dict[str, list[str]] = defaultdict(list)
    for contract_field in core:
        # Markdown task-list syntax, which Streamlit renders as real checkboxes.
        # A bullet plus my own "✓"/"○" produced a doubled marker ("• ○ Edad"), and
        # this is the checkbox the user asked for in the first place.
        if contract_field.name in state.answers:
            value = format_value(contract_field, state.answers[contract_field.name])
            item = f"- [x] {contract_field.label_es} · {value}"
        else:
            item = f"- [ ] {contract_field.label_es}"
        groups[source_group(contract_field) if not contract_field.askable else FROM_MEMORY].append(item)

    # The interview answers lead, then the measurements in contract order, so the
    # panel reads the same way every time.
    ordered = [FROM_MEMORY] + [g for g in groups if g != FROM_MEMORY]

    provided = [f.name for f in core if f.name in state.answers]
    beyond = [name for name in state.answers if name not in set(CORE_FIELDS)]

    with st.container(border=True):
        st.markdown(f"### Tus datos · {len(provided)} de {len(core)}")
        st.progress(len(provided) / len(core))
        for group in ordered:
            st.markdown(f"**{group}**")
            st.markdown("\n".join(groups[group]))
        if beyond:
            st.caption(
                f"Y {len(beyond)} dato(s) más que aportaste sin que los pidiéramos: "
                "cuentan para el resultado igual."
            )


def engine_label(active: Interviewer) -> str:
    if isinstance(active, GeminiInterviewer):
        return "Gemini (entiende respuestas en lenguaje natural)"
    return f"determinista (sin {API_KEY_ENV})"


def main() -> None:
    st.set_page_config(
        page_title="Riesgo cardíaco — demo académico",
        page_icon="🫀",
        layout="wide",
    )
    if load_model() is None:
        return

    if "messages" not in st.session_state:
        start_interview()

    # Everything lives inside the conversation's column instead of running the full
    # width of a projector. At 1350px the disclaimer was a disproportionate pink bar
    # and the engine caption a 145-character line, and neither aligned with the chat
    # underneath it.
    header, _ = st.columns([3, 2], gap="large")
    with header:
        # The typeface is configured natively in .streamlit/config.toml, so this file
        # never injects raw HTML.
        st.error(DISCLAIMER, icon="⚠️")
        st.caption(DISCLAIMER_DETAIL)
        st.title("Riesgo cardíaco")
        st.caption(
            "Cuéntame tus hábitos y, si los tienes, tus números. Al final te muestro "
            "qué tan sólido es el resultado y qué te convendría medir."
        )

        if isinstance(interviewer(), GeminiInterviewer):
            answering = getattr(interviewer(), "model", "")
            st.caption(
                f"Motor de entrevista: {engine_label(interviewer())}"
                + (f" · {answering}" if answering else "")
                + ". Tus respuestas se envían a Google (plan gratuito), que puede "
                "usarlas para mejorar sus modelos; no escribas datos reales de salud."
            )
        else:
            # A missing key is a loud state, not a silent downgrade: the demo still
            # runs, and whoever deployed it needs to know it is running degraded.
            st.warning(
                "**Falta la API key de Gemini.** La app está corriendo con el "
                "entrevistador determinista, que entiende mucho menos. Quien "
                f"despliega esto configura el secreto `{API_KEY_ENV}` en "
                "`.streamlit/secrets.toml` o en la sección de secretos del hosting. "
                "**Los visitantes no necesitan aportar ninguna key.**",
                icon="🔑",
            )
            st.caption(f"Motor de entrevista: {engine_label(interviewer())}")

    result: Prediction | None = st.session_state.result
    if result is not None and st.session_state.interview.finished:
        # The interview is over, so the conversation has nothing left to say: the
        # result takes the screen. The user's read was right, and the transcript is
        # one click away for whoever wants to check what was answered.
        #
        # It lives in the same column as the header rather than running the full
        # width: a 1340px card under a 770px heading is the same misalignment the
        # checklist used to have.
        with st.columns([3, 2], gap="large")[0]:
            render_band(result)
            with st.expander("Ver la conversación"):
                for role, text in st.session_state.messages:
                    with st.chat_message(
                        role, avatar="🫀" if role == "assistant" else None
                    ):
                        st.markdown(text)
            if st.button("Empezar de nuevo", type="primary"):
                start_interview()
                st.rerun()
        return

    # Two columns instead of one narrow strip with a sidebar: the conversation keeps
    # a readable measure, and the width a centred layout wasted now holds the
    # checklist. The chat is the wider one on purpose.
    conversation, panel = st.columns([3, 2], gap="large")

    with panel:
        render_data_checklist(st.session_state.interview)

    with conversation:
        if st.session_state.get("emergency"):
            st.error(EMERGENCY_MESSAGE, icon="🚑")

        for role, text in st.session_state.messages:
            with st.chat_message(role, avatar="🫀" if role == "assistant" else None):
                st.markdown(text)

        if st.session_state.get("emergency"):
            if st.button("Seguir con la entrevista", type="primary"):
                st.session_state.emergency = False
                pending = st.session_state.interview.pending
                follow_up = (
                    question_for(FIELDS_BY_NAME[pending[0]]) if pending else "Seguimos."
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

        prompt = st.chat_input("Escribe tu respuesta…")
        if prompt:
            st.session_state.messages.append(("user", prompt))
            # Streamlit paints the deltas it has already produced, and the transcript
            # above was rendered before this handler ran, so without echoing here the
            # person watches a frozen page for the twenty seconds two model calls take.
            # The echo plus the spinner is the entire feedback budget of a turn.
            with st.chat_message("user", avatar=None):
                st.markdown(prompt)
            with st.chat_message("assistant", avatar="🫀"):
                with st.spinner("Leyendo tu respuesta…"):
                    apply_turn(interviewer().respond(prompt, st.session_state.interview))
            st.rerun()


# Streamlit executes this file as a script, so the app starts unconditionally.
main()
