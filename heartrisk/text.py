"""One text-folding helper shared by the interview and the safety check.

It lives on its own so that ``safety`` and ``interview`` can both use it without
importing each other: the interview has to consult the safety check, so the
safety check cannot import the interview.
"""

from __future__ import annotations

import re

_ACCENTS = str.maketrans(
    {
        "á": "a",
        "é": "e",
        "í": "i",
        "ó": "o",
        "ú": "u",
        "ü": "u",
        "ñ": "n",
    }
)


def normalise(text: str) -> str:
    """Lowercase, strip accents and collapse whitespace for keyword matching."""
    return " ".join(text.lower().translate(_ACCENTS).split())


def tokens_of(text: str) -> set[str]:
    return set(normalise(text).split())


def contains_any(folded: str, phrases: tuple[str, ...]) -> bool:
    return any(phrase in folded for phrase in phrases)


#: A number is only a number when a letter does not introduce it. Without this,
#: "mi hba1c es 5,4" yields the 1 inside "hba1c".
NUMBER = re.compile(r"(?<![A-Za-z\u00e1\u00e9\u00ed\u00f3\u00fa\u00f1])-?\d+(?:[.,]\d+)*")


def parse_number(text: str) -> float | None:
    """Read the first number in Spanish formatting, or ``None``.

    ``"5,4"`` is 5.4 and ``"10,000"`` is 10000. A Spanish thousands separator
    written with a dot (``"1.200"``) reads as 1.2; the contract's plausibility
    bounds catch the absurd result and the question gets re-asked.
    """
    match = NUMBER.search(text)
    if match is None:
        return None
    raw = match.group(0)
    if "," in raw and "." not in raw:
        head, _, tail = raw.rpartition(",")
        raw = head.replace(",", "") + tail if len(tail) == 3 else f"{head}.{tail}"
    return float(raw.replace(",", ""))


def parse_numbers(text: str) -> list[float]:
    """Every number in the text, in order, read with Spanish conventions."""
    values: list[float] = []
    for match in NUMBER.finditer(text):
        value = parse_number(match.group(0))
        if value is not None:
            values.append(value)
    return values
