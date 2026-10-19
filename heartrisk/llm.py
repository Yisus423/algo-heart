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
from dataclasses import dataclass, field
from typing import Protocol

from .interview import (
    Interviewer,
    InterviewState,
    StubInterviewer,
    Turn,
    emergency_turn,
    is_skip,
    parse_answer,
    question_for,
)
from .schema import (
    FIELDS,
    FIELDS_BY_NAME,
    Field,
    SchemaError,
    to_native,
    validate_value,
)

DEFAULT_MODEL = "gemini-2.5-flash"
API_KEY_ENV = "GEMINI_API_KEY"

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
    "Tu tarea es entender lo que la persona escribe y reportarlo con la función "
    f"`{TOOL_NAME}`. No decides qué preguntar: el sistema te da la próxima "
    "pregunta y tú la formulas de forma natural.\n\n"
    "Reglas:\n"
    "- Hablas en español neutro, sin voseo, breve y cálido. Una o dos frases.\n"
    "- Nunca inventes un valor. Si la persona no dijo algo, no lo reportes.\n"
    "- Si dijo algo aproximado ('como 95 kilos', 'más o menos 3000 pasos'), "
    "reporta el número que dio.\n"
    "- Si dijo que no sabe o no tiene el dato, no reportes ese campo.\n"
    "- NUNCA digas que la persona puede tener una enfermedad, ni des un "
    "diagnóstico, ni sugieras un porcentaje de riesgo. Eres un demo, no un médico.\n"
    "- Si la persona describe un síntoma agudo (dolor de pecho ahora, falta de "
    "aire, desmayo), no sigas con la entrevista: dile que llame a emergencias.\n"
    "- No pidas datos que no estén en la lista de campos.\n"
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
        self, system: str, contents: list[dict[str, object]], tool: dict[str, object]
    ) -> ModelReply:
        """Send the conversation and the tool definition, return one reply."""
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
        self, system: str, contents: list[dict[str, object]], tool: dict[str, object]
    ) -> ModelReply:
        from google import genai
        from google.genai import types

        declaration = types.FunctionDeclaration(
            name=str(tool["name"]),
            description=str(tool["description"]),
            parameters_json_schema=tool["parameters_json_schema"],
        )
        client = genai.Client(api_key=self._api_key)
        response = client.models.generate_content(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=self.temperature,
                tools=[types.Tool(function_declarations=[declaration])],
            ),
        )

        arguments: dict[str, object] = {}
        for call in response.function_calls or []:
            if call.name == TOOL_NAME and call.args:
                arguments.update(dict(call.args))
        return ModelReply(text=(response.text or "").strip(), arguments=arguments)


class GeminiInterviewer:
    """Natural-language understanding on top of the contract's fixed agenda."""

    def __init__(self, transport: Transport) -> None:
        self._transport = transport
        self._transcript: list[dict[str, object]] = []
        self._fallback = StubInterviewer()

    def opening(self) -> Turn:
        """The deterministic first turn, so startup never waits on the network."""
        return self._fallback.opening()

    def respond(self, text: str, state: InterviewState) -> Turn:
        emergency = emergency_turn(text)
        if emergency is not None:
            return emergency

        self._transcript.append({"role": "user", "parts": [{"text": text}]})
        pending = FIELDS_BY_NAME.get(state.pending) if state.pending else None

        reply = self._transport.generate(
            SYSTEM_PROMPT, list(self._transcript), build_answer_tool()
        )
        if reply.text:
            self._transcript.append({"role": "model", "parts": [{"text": reply.text}]})

        rejected = self._store(reply.arguments, state)

        if pending is not None and pending.name not in state.attempted:
            if is_skip(text):
                state.skipped.add(pending.name)
            else:
                try:
                    state.answers[pending.name] = parse_answer(pending, text)
                except SchemaError:
                    pass

        messages = [reply.text] if reply.text else []
        if rejected:
            messages.append(
                "No pude usar alguno de esos valores, así que no los guardé: "
                + "; ".join(rejected)
            )

        if pending is not None and pending.name not in state.attempted:
            # The model did not resolve the field being asked and the fallback
            # parser could not either: ask it again instead of moving on.
            state.pending = pending.name
            messages.append(f"Perdón, de nuevo: {question_for(pending)}")
            return Turn(messages=tuple(messages), asking=pending.name)

        following = state.next_field()
        if following is None:
            state.pending = None
            state.finished = True
            messages.append("Perfecto, con esto ya puedo calcular el resultado.")
            return Turn(messages=tuple(messages), complete=True)

        state.pending = following.name
        messages.append(question_for(following))
        return Turn(messages=tuple(messages), asking=following.name)

    def _store(
        self, arguments: dict[str, object], state: InterviewState
    ) -> list[str]:
        """Validate reported values and keep only the ones the contract accepts.

        A rejected value is reported back to the user rather than dropped in
        silence, because a silent drop looks like the program ignored them.
        """
        rejected: list[str] = []
        for name, value in arguments.items():
            contract_field = FIELDS_BY_NAME.get(name)
            if contract_field is None or contract_field.computed_from:
                continue
            if name in state.answers:
                continue
            native = to_native(value)
            try:
                validate_value(contract_field, native)
            except SchemaError as exc:
                rejected.append(str(exc))
                continue
            state.answers[name] = native
        return rejected


def build_interviewer(api_key: str | None = None, model: str = DEFAULT_MODEL) -> Interviewer:
    """Use Gemini when a key is available, otherwise the deterministic stub.

    Never raises: an app that cannot reach a language model must still run, and
    the deterministic interviewer collects the same contract fields.
    """
    key = api_key or os.environ.get(API_KEY_ENV)
    if not key:
        return StubInterviewer()
    try:
        return GeminiInterviewer(GeminiTransport(model=model, api_key=key))
    except RuntimeError:
        return StubInterviewer()


__all__ = [
    "API_KEY_ENV",
    "DEFAULT_MODEL",
    "GeminiInterviewer",
    "GeminiTransport",
    "Interviewer",
    "ModelReply",
    "TOOL_NAME",
    "Transport",
    "build_answer_tool",
    "build_interviewer",
    "field_schema",
]
