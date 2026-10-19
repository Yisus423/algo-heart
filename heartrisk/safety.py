"""Acute-symptom detection, and the message that replaces a prediction.

Why this is separate from the model
-----------------------------------
The model answers a question about a synthetic dataset. It has nothing useful to
say to someone describing chest pain that started twenty minutes ago, and
answering that person with a probability would be worse than saying nothing.

Design bias
-----------
The rules lean towards firing. A false positive costs the user a message telling
them to call emergency services if this is happening now, and the interview can
continue afterwards; a false negative means the demo keeps asking about sleep
habits while someone describes a heart attack. The bias is therefore towards
over-triggering, and it is stated here so nobody later "optimises" it away.

Two rule shapes:

* Unambiguous acute presentations fire on their own ("no puedo respirar", "me
  desmayé", "dolor que baja al brazo").
* Ambiguous ones need a present-tense marker as well ("dolor en el pecho" alone
  is the ordinary interview answer, so it needs "ahora" or "hace un rato").
"""

from __future__ import annotations

from .text import contains_any, normalise

#: Sufficient on their own: these describe an acute event, not a habit.
#:
#: Deliberately NOT here, because each one would fire on an ordinary interview
#: answer: "infarto" and "ataque al corazón" (the family-history question asks
#: exactly that about a relative), "arritmia" and "taquicardia" (diagnosed
#: conditions people mention in passing), and "me falta el aire" (which is what
#: exertional breathlessness sounds like). Those live in SYMPTOM_PHRASES and need
#: a present-tense marker.
ACUTE_PHRASES: tuple[str, ...] = (
    "no puedo respirar",
    "no puede respirar",
    "no puedo respirar bien",
    "no me entra aire",
    "me desmaye",
    "me desmayo",
    "me desvaneci",
    "me estoy por desmayar",
    "perdi el conocimiento",
    "dolor que baja al brazo",
    "dolor al brazo izquierdo",
    "dolor que se corre al brazo",
    "me aprieta el pecho",
    "opresion en el pecho",
    "apretado en el pecho",
    "sudor frio",
    "labios azules",
    "se me nublo la vista",
    "creo que estoy teniendo un infarto",
    "estoy teniendo un infarto",
    "me estoy muriendo",
)

#: Ordinary or ambiguous symptoms. They only fire together with a present-tense
#: marker, because they are also the expected answers to the interview's own
#: questions.
SYMPTOM_PHRASES: tuple[str, ...] = (
    "dolor en el pecho",
    "dolor de pecho",
    "dolor al pecho",
    "presion en el pecho",
    "peso en el pecho",
    "molestia en el pecho",
    "me duele el pecho",
    "falta de aire",
    "me falta el aire",
    "me quede sin aire",
    "dificultad para respirar",
    "me mareo",
    "mareos",
    "nauseas",
    "palpitaciones",
    "taquicardia",
    "arritmia",
    "convulsiones",
    "infarto",
    "ataque al corazon",
)

#: Present-tense markers. "hace un rato" and "desde hace" are included because a
#: recent onset is still an emergency, not a historical habit.
NOW_MARKERS: tuple[str, ...] = (
    "ahora",
    "en este momento",
    "hace un rato",
    "hace unos minutos",
    "recien",
    "esta pasando",
    "me esta pasando",
    "desde hace",
    "de repente",
    "subito",
    "repentino",
    "en este instante",
)

#: Shown instead of any prediction. Spanish, because it is user-facing copy.
EMERGENCY_MESSAGE = (
    "**Esto suena a algo que necesita atención ahora, no un demo.**\n\n"
    "Si estás sintiendo dolor u opresión en el pecho, te falta el aire, te "
    "desmayaste o el dolor te baja por el brazo en este momento, **detente aquí y "
    "llama a emergencias** (en Venezuela, 911) o acude a una sala de "
    "emergencias.\n\n"
    "Este programa no puede evaluar nada de eso y no es un servicio médico."
)


def detect_emergency(text: str) -> str | None:
    """Return why the text looks like an acute emergency, or ``None``.

    The return value is a short Spanish reason, for logging or display in a
    caption; the caller shows :data:`EMERGENCY_MESSAGE` to the user.
    """
    folded = normalise(text)
    if not folded:
        return None

    for phrase in ACUTE_PHRASES:
        if phrase in folded:
            return f"menciona {phrase!r}"

    if contains_any(folded, SYMPTOM_PHRASES) and contains_any(folded, NOW_MARKERS):
        return "describe un síntoma agudo en presente"

    return None
