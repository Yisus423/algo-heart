"""Spec S9: the copy is neutral Spanish.

The user's correction: "usaste el español argentino de tu persona en algo que
deberia ser neutro, sin voseo ni nada, sería super raro llevar eso asi estando en
venezuela".

The app was written in Rioplatense Spanish, with voseo and Argentina-only
references, for a Venezuelan audience. That is a product defect, so it gets a test
rather than a promise: every user-facing string in the project is extracted from
the source and checked for the second-person forms and regional words that must
not come back.

Docstrings are excluded: they are written for maintainers, in English, and are not
copy. The keyword cue lists in `safety.py` and `interview.py` are input parsers
rather than copy, and they are accent-stripped and dialect-neutral by construction.
"""

from __future__ import annotations

import ast
import pathlib
import re

import pytest

from heartrisk.llm import SYSTEM_PROMPT
from heartrisk.safety import EMERGENCY_MESSAGE
from heartrisk.schema import FIELDS

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

USER_FACING_MODULES = (
    "app.py",
    "heartrisk/schema.py",
    "heartrisk/interview.py",
    "heartrisk/safety.py",
    "heartrisk/llm.py",
)

#: Every voseo form is written out WITH its accent. The accent is the only thing
#: separating voseo from the neutral second person ("pesás" from "pesas", "hacés"
#: from "haces", "registrá" from "registra"), so a character class like `pes[aá]s`
#: flags the correct neutral form instead. `IGNORECASE` is only for a capital at the
#: start of a sentence. "voseo" itself is not on the list: the system prompt names
#: it in order to forbid it, and flagging that would be the check shouting at its
#: own rule.
VOSEO = re.compile(
    r"\b(tenés|podés|sabés|sentís|fumás|dormís|usás|caminás|pesás|medís|hacés|"
    r"querés|preferís|vivís|formulás|sos|decime|contame|respondeme|fijate|"
    r"acordate|mirá|escribí|empezá|cortá|llamá|andá|registrá|traducí|revisá|"
    r"probá|poné|acá|guardia|CABA|rioplatense)\b",
    re.IGNORECASE,
)


def _docstrings(tree: ast.AST) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            text = ast.get_docstring(node, clean=False)
            if text is not None:
                found.add(text)
    return found


def user_facing_strings(relative_path: str) -> list[tuple[int, str]]:
    tree = ast.parse((REPO_ROOT / relative_path).read_text())
    docstrings = _docstrings(tree)
    return [
        (node.lineno, node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value not in docstrings
    ]


@pytest.mark.parametrize("relative_path", USER_FACING_MODULES)
def test_no_user_facing_string_uses_voseo_or_a_regional_word(relative_path: str) -> None:
    offenders = [
        (line, match.group(0), text[:80])
        for line, text in user_facing_strings(relative_path)
        if (match := VOSEO.search(text))
    ]

    assert not offenders, f"voseo still present in {relative_path}: {offenders}"


def test_the_questions_use_the_neutral_second_person() -> None:
    """The measurement questions are the ones that read as second person."""
    hdl_question = FIELDS[[f.name for f in FIELDS].index("hdl")].question_es

    assert "¿Tienes" in hdl_question


def test_the_interviewer_opens_in_a_neutral_register() -> None:
    from heartrisk.interview import StubInterviewer

    opening = " ".join(StubInterviewer().opening().messages)

    assert "Voy a hacerte" in opening
    assert "Puedes responder" in opening


def test_the_emergency_message_names_no_argentina_only_number() -> None:
    assert "CABA" not in EMERGENCY_MESSAGE
    assert "107" not in EMERGENCY_MESSAGE
    assert "sala de emergencias" in EMERGENCY_MESSAGE


def test_the_model_is_told_which_register_to_write_in() -> None:
    """The language model writes most of the visible copy, so it is told the rule."""
    assert "español neutro" in SYSTEM_PROMPT
    assert "sin voseo" in SYSTEM_PROMPT


def test_no_question_uses_the_rioplatense_imperative() -> None:
    for contract_field in FIELDS:
        assert not VOSEO.search(contract_field.question_es), contract_field.name
        if contract_field.needs_es:
            assert not VOSEO.search(contract_field.needs_es), contract_field.name
