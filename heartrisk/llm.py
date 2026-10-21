"""The language-model interviewer.

The decision that shapes this module
------------------------------------
The model owns *understanding*; the contract owns the *agenda*.

The model reads whatever the person actually wrote — "tengo como 95 kilos y mido
1,72", "fumo desde los 20", "camino poco" — and reports what it understood
through a tool call. It does not decide what to ask next. The next question
always comes from :mod:`heartrisk.schema`, in a fixed order, chosen by
:meth:`heartrisk.interview.InterviewState.next_field`.

That split is deliberate. An interviewer that lets the model drive the agenda can
loop, can forget a field, can invent a question, and cannot be tested without a
network. Here the model can only add understanding, and the interview always
advances.

Robustness, since a demo must not die on stage
---------------------------------------------
* The first turn is deterministic, so nothing blocks on a network call before the
  user has typed anything.
* Deterministic parsing runs as a fallback for the field being asked, so an
  unavailable or dull model still collects data.
* Whatever the model reports is validated against the contract. A value that
  fails is not stored; the field is asked again.
* :func:`heartrisk.interview.emergency_turn` runs before any model call.

Environment: ``GEMINI_API_KEY``. Model: ``gemini-2.5-flash`` unless overridden.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

from .interview import (
    Interviewer,
    InterviewState,
    StubInterviewer,
    Turn,
    emergency_turn,
    batch_question,
    is_skip,
    parse_answer,
    question_for,
)
from .schema import (
    FIELDS,
    FIELDS_BY_NAME,
    Field,
    SchemaError,
    rejection_message,
    to_native,
    validate_value,
)

DEFAULT_MODEL = "gemini-2.5-flash"
API_KEY_ENV = "GEMINI_API_KEY"

#: Said by the app, not by the model. Gemini returns no text alongside a tool
#: call, so leaving the acknowledgement to it produced a mute interviewer, and
#: asking it for a sentence produced a second question in a different register.
ACKNOWLEDGEMENT = "Anotado, gracias."

#: Used for the second, tool-less call of a turn. Verified against the raw
#: response: when Gemini calls a function the reply contains only that call and no
#: text part at all, so the phrasing cannot ride along with the extraction.
PHRASING_PROMPT = (
    "Reformulas preguntas para una persona, en español neutro y tratándola de "
    "'tú'. Pueden ser una o varias. Devuelves únicamente el mensaje con las "
    "preguntas, corto y cálido, sin responderlas y sin agregar nada más."
)

#: Accepted spellings, looked up in order. Streamlit's own documentation uses the
#: upper-case form; Python developers tend to write the lower-case one.
API_KEY_NAMES: tuple[str, ...] = ("GEMINI_API_KEY", "gemini_api_key")
MODEL_NAMES: tuple[str, ...] = ("GEMINI_MODEL", "gemini_model")

TOOL_NAME = "registrar_datos"
TOOL_DESCRIPTION = (
    "Registra los datos que la persona mencionó en su mensaje, aunque los haya "
    "dicho de forma informal o aproximada. Llama a esta función SOLO con los "
    "campos que la persona efectivamente dijo, y traduce lo que dijo al valor "
    "que corresponde. Nunca inventes un valor y nunca completes un campo con un "
    "número típico: dejar un campo afuera es correcto, inventarlo no lo es."
)

SYSTEM_PROMPT = (
    "Eres el entrevistador de un demo académico que estima riesgo cardíaco con un "
    "modelo entrenado sobre datos SINTÉTICOS.\n\n"
    "Tu tarea es entender lo que la persona cuenta y registrar cada dato con la "
    f"función `{TOOL_NAME}`. Las preguntas las maneja el sistema.\n\n"
    "Reglas:\n"
    "- Hablas en español neutro y tratas a la persona de 'tú': nunca 'usted' y "
    "nunca voseo.\n"
    "- Nunca inventes un valor. Si la persona no dijo algo, no lo reportes.\n"
    "- Si dijo algo aproximado ('como 95 kilos', 'más o menos 3000 pasos'), "
    "reporta el número que dio.\n"
    "- Si dijo que no sabe o no tiene el dato, no reportes ese campo.\n"
    "- NUNCA digas que la persona puede tener una enfermedad, ni des un "
    "diagnóstico, ni sugieras un porcentaje de riesgo. Eres un demo, no un médico.\n"
    "- Si la persona describe un síntoma agudo (dolor de pecho ahora, falta de "
    "aire, desmayo), no sigas con la entrevista: dile que llame a emergencias.\n"
)


@dataclass(frozen=True, slots=True)
class ModelReply:
    """What one model call came back with.

    Attributes:
        text: the model's prose, empty when it only called the tool.
        arguments: the tool arguments it produced, empty when it only spoke.
    """

    text: str = ""
    arguments: dict[str, object] = field(default_factory=dict)


class Transport(Protocol):
    """The network boundary, so tests can run without one."""

    def generate(
        self, system: str, contents: list[dict[str, object]], tool: dict[str, object] | None
    ) -> ModelReply:
        """Send the conversation and, optionally, the tool definition.

        ``tool`` is ``None`` for the phrasing call, which must not be able to make
        a tool call because that is exactly what suppresses its text.
        """
        ...


def field_schema(contract_field: Field) -> dict[str, object]:
    """Describe one contract field as JSON Schema for the tool call.

    The tool schema is the contract, generated from it rather than written out
    again, so a field can never be askable in the interview but missing from what
    the model is allowed to report.
    """
    description = contract_field.label_es
    if contract_field.unit_es:
        description += f" ({contract_field.unit_es})"
    if contract_field.needs_es:
        description += f". Sale de {contract_field.needs_es}"
    if contract_field.computed_from:
        description += f". Se calcula a partir de {', '.join(contract_field.computed_from)}"

    schema: dict[str, object] = {"description": description}
    if contract_field.kind == "numeric":
        schema["type"] = "number"
        if contract_field.minimum is not None:
            schema["minimum"] = contract_field.minimum
        if contract_field.maximum is not None:
            schema["maximum"] = contract_field.maximum
    elif contract_field.kind == "boolean":
        schema["type"] = "boolean"
    else:
        schema["type"] = "string"
        schema["enum"] = list(contract_field.categories)
    return schema


def build_answer_tool() -> dict[str, object]:
    """The single tool the model may call."""
    return {
        "name": TOOL_NAME,
        "description": TOOL_DESCRIPTION,
        "parameters_json_schema": {
            "type": "object",
            "properties": {
                contract_field.name: field_schema(contract_field)
                for contract_field in FIELDS
                if not contract_field.computed_from
            },
            "additionalProperties": False,
        },
    }


def _response_text(response: object) -> str:
    """Read the text parts of a response that also carries a tool call.

    The SDK's ``response.text`` helper logs a warning and concatenates only the
    text parts whenever the response contains anything else — which is every reply
    that calls the tool. Reading the parts directly keeps the console clean and
    makes the intent explicit.
    """
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return ""
    content = getattr(candidates[0], "content", None)
    parts = getattr(content, "parts", None) or []
    return " ".join(
        part.text for part in parts if getattr(part, "text", None)
    ).strip()


class GeminiTransport:
    """``google-genai`` implementation of :class:`Transport`.

    The import is deferred so that importing this module, and therefore running
    the app with the deterministic interviewer, does not require the SDK.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        temperature: float = 0.2,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self._api_key = api_key or os.environ.get(API_KEY_ENV)
        if not self._api_key:
            raise RuntimeError(
                f"{API_KEY_ENV} is not set; export it, or use the deterministic "
                "interviewer instead"
            )

    def generate(
        self, system: str, contents: list[dict[str, object]], tool: dict[str, object] | None
    ) -> ModelReply:
        from google import genai
        from google.genai import types

        tools = None
        if tool is not None:
            tools = [
                types.Tool(
                    function_declarations=[
                        types.FunctionDeclaration(
                            name=str(tool["name"]),
                            description=str(tool["description"]),
                            parameters_json_schema=tool["parameters_json_schema"],
                        )
                    ]
                )
            ]
        client = genai.Client(api_key=self._api_key)
        response = client.models.generate_content(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=self.temperature,
                tools=tools,
            ),
        )

        arguments: dict[str, object] = {}
        for call in response.function_calls or []:
            if call.name == TOOL_NAME and call.args:
                arguments.update(dict(call.args))
        return ModelReply(text=_response_text(response), arguments=arguments)


class GeminiInterviewer:
    """Natural-language understanding on top of the contract's fixed agenda."""

    def __init__(self, transport: Transport) -> None:
        self._transport = transport
        self._transcript: list[dict[str, object]] = []
        self._fallback = StubInterviewer()

    def opening(self) -> Turn:
        """The free-form invitation, which only this interviewer can act on.

        The stub cannot use it: turning "tengo 62, peso 95 y fumo" into three fields
        takes understanding, so the deterministic interviewer opens with its first
        question instead. Here it is the whole point, because it is what makes the
        interview two turns long for somebody who writes, instead of twenty-five.

        It makes no network call, so startup never waits on the model.
        """
        return Turn(
            messages=(
                "¡Hola! Soy el asistente de este demo.",
                "Cuéntame lo que sepas de ti: tu edad, tus hábitos, y si tienes "
                "análisis a mano, los números. Puedes escribirlo todo junto y en el "
                "orden que quieras.",
                "Si algo no lo sabes, dime \"no sé\" y seguimos. Y cuando quieras "
                "cortar, escribe \"listo\".",
            )
        )

    def respond(self, text: str, state: InterviewState) -> Turn:
        emergency = emergency_turn(text)
        if emergency is not None:
            return emergency

        self._transcript.append({"role": "user", "parts": [{"text": text}]})
        pending = FIELDS_BY_NAME.get(state.pending[0]) if state.pending else None

        try:
            reply = self._transport.generate(
                SYSTEM_PROMPT, list(self._transcript), build_answer_tool()
            )
        except Exception:
            # The model is a third party. On stage it can rate-limit or return a
            # 503, and a traceback in front of an audience is worse than a plainer
            # interview, so the turn degrades instead of crashing.
            return self._offline_turn(text, state, pending)

        rejected, registered = self._store(reply.arguments, state)

        if pending is not None and pending.name not in state.attempted:
            if is_skip(text):
                state.skipped.add(pending.name)
            else:
                try:
                    state.answers[pending.name] = parse_answer(pending, text)
                except SchemaError:
                    pass

        messages: list[str] = []
        if rejected:
            messages.append(
                "No pude usar alguno de esos valores, así que no los guardé: "
                + "; ".join(rejected)
            )

        if pending is not None and pending.name not in state.attempted:
            # Neither the model nor the fallback parser resolved the field that was
            # asked, so ask it again rather than moving on and losing it.
            state.pending = (pending.name,)
            messages.append(f"Perdón, de nuevo: {question_for(pending)}")
            return Turn(messages=tuple(messages), asking=(pending.name,))

        following = state.next_batch()
        if not following:
            state.pending = ()
            state.finished = True
            messages.append("Perfecto, con esto ya puedo calcular el resultado.")
            return Turn(messages=tuple(messages), complete=True)

        # The contract decided which fields are worth asking; the model decides how
        # to ask them, and a batch of them arrives as a single message.
        state.pending = tuple(f.name for f in following)
        if registered or rejected:
            messages.append(ACKNOWLEDGEMENT)
        messages.append(self._phrase(following))
        return Turn(messages=tuple(messages), asking=state.pending)

    def _phrase(self, fields: Sequence[Field]) -> str:
        """Put the contract's questions in the model's words, in their own call.

        Separate because Gemini returns no text alongside a tool call. If it fails,
        or comes back without a question mark, the contract's own wording is used, so
        the interview never leaves the person with nothing to answer.
        """
        try:
            # The instruction must be explicit that these are questions to ask of
            # someone else. Sending a bare question as a user turn made the model
            # read it as addressed to itself: asked to rephrase "¿Cuál es tu sexo?"
            # it answered "¿Cómo te gustaría que me identificara?".
            questions = "\n".join(f"- {question_for(f, hint=False)}" for f in fields)
            instruction = (
                "Hazle estas preguntas a la persona, todas juntas en un solo "
                "mensaje, sin responderlas, y cierra recordándole una sola vez que "
                f"puede decir \"no sé\" si no las sabe:\n\n{questions}"
            )
            reply = self._transport.generate(
                PHRASING_PROMPT,
                [{"role": "user", "parts": [{"text": instruction}]}],
                None,
            )
        except Exception:
            return batch_question(fields)
        phrased = reply.text.strip()
        return phrased if "?" in phrased else batch_question(fields)

    def _offline_turn(
        self, text: str, state: InterviewState, pending: Field | None
    ) -> Turn:
        """Keep the interview moving when the model cannot be reached at all.

        The deterministic parser still handles the field being asked, so the answer
        is not lost, and the person is told plainly that the model is down.
        """
        if pending is not None and pending.name not in state.attempted:
            if is_skip(text):
                state.skipped.add(pending.name)
            else:
                try:
                    state.answers[pending.name] = parse_answer(pending, text)
                except SchemaError:
                    pass

        messages = [
            "Se me cortó la conexión con el modelo. Guardé tu respuesta de la forma "
            "más simple que pude y seguimos igual."
        ]

        if pending is not None and pending.name not in state.attempted:
            state.pending = (pending.name,)
            messages.append(f"Perdón, de nuevo: {question_for(pending)}")
            return Turn(messages=tuple(messages), asking=(pending.name,))

        # One question at a time here, not a batch: only the model can match several
        # free-text answers to several questions, and the deterministic parser would
        # file "62" under whichever field it happened to try. Slower and correct
        # beats fast and wrong.
        following = state.next_field()
        if following is None:
            state.pending = ()
            state.finished = True
            messages.append("Con esto ya puedo calcular el resultado.")
            return Turn(messages=tuple(messages), complete=True)

        state.pending = (following.name,)
        messages.append(question_for(following))
        return Turn(messages=tuple(messages), asking=(following.name,))

    def _store(
        self, arguments: dict[str, object], state: InterviewState
    ) -> tuple[list[str], int]:
        """Validate reported values and keep only the ones the contract accepts.

        A rejected value is reported back to the user rather than dropped in
        silence, because a silent drop looks like the program ignored them.

        Returns the rejection messages and how many values were kept, the latter
        so the caller can acknowledge only what was actually registered.
        """
        rejected: list[str] = []
        stored = 0
        for name, value in arguments.items():
            contract_field = FIELDS_BY_NAME.get(name)
            if contract_field is None or contract_field.computed_from:
                continue
            if name in state.answers:
                continue
            native = to_native(value)
            try:
                validate_value(contract_field, native)
            except SchemaError:
                rejected.append(rejection_message(contract_field, native))
                continue
            state.answers[name] = native
            stored += 1
        return rejected, stored


def _lookup(sources: list[object], names: tuple[str, ...]) -> str | None:
    """First non-blank string found under any of ``names``, in source order."""
    for source in sources:
        if not source:
            continue
        for name in names:
            value = source.get(name)  # type: ignore[union-attr]
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def resolve_api_key(
    secrets: object | None = None, env: object | None = None
) -> str | None:
    """Return the API key, preferring the deployment's secrets over the environment.

    Secrets come first because the deployment owns this credential: a visitor
    brings nothing of their own, and a stray local ``export`` must not silently
    override what the deployed app was configured with.
    """
    return _lookup([secrets, os.environ if env is None else env], API_KEY_NAMES)


def resolve_model(
    secrets: object | None = None, env: object | None = None, default: str = DEFAULT_MODEL
) -> str:
    """Return the configured model name, falling back to ``default``."""
    return _lookup([secrets, os.environ if env is None else env], MODEL_NAMES) or default


def build_interviewer(
    api_key: str | None = None, model: str | None = None, secrets: object | None = None
) -> Interviewer:
    """Use Gemini when a key is available, otherwise the deterministic stub.

    Never raises: an app that cannot reach a language model must still run, and
    the deterministic interviewer collects the same contract fields. The caller
    decides how loudly to say that the model is absent.
    """
    key = api_key or resolve_api_key(secrets)
    if not key:
        return StubInterviewer()
    try:
        return GeminiInterviewer(
            GeminiTransport(model=model or resolve_model(secrets), api_key=key)
        )
    except RuntimeError:
        return StubInterviewer()


__all__ = [
    "API_KEY_ENV",
    "API_KEY_NAMES",
    "DEFAULT_MODEL",
    "MODEL_NAMES",
    "GeminiInterviewer",
    "GeminiTransport",
    "Interviewer",
    "ModelReply",
    "TOOL_NAME",
    "Transport",
    "build_answer_tool",
    "build_interviewer",
    "field_schema",
    "resolve_api_key",
    "resolve_model",
]
