"""UI-level guards.

The typeface is configured through Streamlit's own theme instead of being injected
as a CSS rule. An earlier version did this:

    st.markdown('<style>[class*="st-"] { font-family: "Nunito" }</style>',
                unsafe_allow_html=True)

That selector also matched Streamlit's icon elements, so it replaced the
"Material Symbols Rounded" font Streamlit ships locally and every Material icon in
the interface stopped rendering. The replacement is native, which is why these
tests exist: a blanket CSS override is easy to reintroduce and hard to notice.
"""

from __future__ import annotations

import pathlib
import re
import tomllib

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
APP_SOURCE = (REPO_ROOT / "app.py").read_text()
THEME = tomllib.loads((REPO_ROOT / ".streamlit" / "config.toml").read_text())["theme"]

#: Selectors broad enough to reach Streamlit's own chrome, not just our content.
BROAD_SELECTORS = (
    r'\[class\*=\s*"st-"\]',
    r'\[class\^=\s*"st-"\]',
    r"\[class\|=\s*\"st-\"\]",
    r"html\s*,\s*body",
)


def test_the_app_injects_no_raw_html() -> None:
    assert "unsafe_allow_html" not in APP_SOURCE
    assert "<style>" not in APP_SOURCE


def test_the_app_never_sets_a_font_family() -> None:
    """Fonts belong to the theme; app.py should not know about them at all."""
    assert "font-family" not in APP_SOURCE


def test_no_broad_selector_is_used_anywhere_in_the_app() -> None:
    offenders = [p for p in BROAD_SELECTORS if re.search(p, APP_SOURCE)]
    assert not offenders, f"broad selectors in app.py: {offenders}"


def test_the_typeface_is_declared_in_the_theme() -> None:
    assert THEME["font"].startswith("Nunito")
    assert THEME["headingFont"].startswith("Nunito")


def test_the_theme_declares_font_files_for_both_styles() -> None:
    faces = THEME["fontFaces"]

    assert {face["style"] for face in faces} == {"normal", "italic"}
    assert all(face["family"] == "Nunito" for face in faces)
    assert all(face["url"].startswith("https://fonts.gstatic.com/") for face in faces)


def test_the_font_keys_live_in_the_theme_table() -> None:
    """A key below [[theme.fontFaces]] belongs to that array element, not [theme].

    TOML has no way to warn about this: the file still parses, the key just stops
    applying. Hence a test rather than a comment.
    """
    assert "font" in THEME
    assert "headingFont" in THEME
    assert "font" not in (THEME["fontFaces"][0] or {})
