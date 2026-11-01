"""Persist one chat session as a JSON file.

A session survives a browser reload and a Streamlit rerun because its state is
written to disk instead of living only in ``st.session_state``. The real
limitation is deployment durability: Streamlit Community Cloud has an ephemeral
filesystem, so a session does not survive a redeploy or a restart there.

Each session has an opaque identifier and one JSON file. Writes are atomic: the
new document is written to a temporary file in the same directory and then
moved into place, so a process that dies mid-write cannot leave a half-written
conversation behind.
"""

from __future__ import annotations

import json
import os
import secrets
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SESSIONS_DIR = REPO_ROOT / "data" / "sessions"

#: A token from ``secrets.token_urlsafe()`` is comfortably below this bound.
MAX_SESSION_ID_LENGTH = 64

#: The on-disk envelope version. Files with another version are ignored.
_STORAGE_VERSION = 1


def new_session_id() -> str:
    """Return a new opaque, URL-safe session identifier."""
    return secrets.token_urlsafe()


def is_valid_session_id(value: object) -> bool:
    """Whether ``value`` is safe to use as a session identifier.

    Only the characters produced by :func:`new_session_id` and a bounded length
    are accepted. In particular, path separators and dots are rejected so an
    identifier arriving from a URL cannot escape the session directory.
    """
    if not isinstance(value, str):
        return False
    if not value or len(value) > MAX_SESSION_ID_LENGTH:
        return False
    return all(
        character.isascii()
        and (character.isalnum() or character in "-_")
        for character in value
    )


def save(
    session_id: str,
    payload: dict[str, object],
    directory: Path | str = DEFAULT_SESSIONS_DIR,
) -> None:
    """Save ``payload`` for ``session_id`` atomically.

    The payload must be a plain dictionary containing only JSON-serializable
    values. Unsupported values are rejected before the session directory or any
    temporary file is created.
    """
    _require_valid_session_id(session_id)
    if not isinstance(payload, dict):
        raise TypeError("session payload must be a plain dictionary")

    document = {"version": _STORAGE_VERSION, "payload": payload}
    try:
        encoded = json.dumps(document, ensure_ascii=False, allow_nan=False)
    except TypeError as exc:
        raise TypeError(
            f"session payload must contain only JSON-serializable values: {exc}"
        ) from exc
    except ValueError as exc:
        raise ValueError(
            f"session payload must contain only JSON-serializable values: {exc}"
        ) from exc

    session_directory = Path(directory)
    session_directory.mkdir(parents=True, exist_ok=True)
    target = session_directory / f"{session_id}.json"
    temporary = session_directory / (
        f".{session_id}.{secrets.token_hex(16)}.tmp"
    )
    temporary_created = False

    try:
        with temporary.open("x", encoding="utf-8") as handle:
            temporary_created = True
            handle.write(encoded)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if temporary_created:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def load(
    session_id: str,
    directory: Path | str = DEFAULT_SESSIONS_DIR,
) -> dict[str, object] | None:
    """Load a saved session, or return ``None`` when it cannot be loaded.

    This function never raises. A missing, corrupt, unreadable, or older-format
    file is treated as an unavailable session so a live demo can start a new
    conversation instead of showing a traceback.
    """
    try:
        if not is_valid_session_id(session_id):
            return None
        path = Path(directory) / f"{session_id}.json"
        with path.open("r", encoding="utf-8") as handle:
            document = json.load(handle)

        if not isinstance(document, dict):
            return None
        version = document.get("version")
        if type(version) is not int or version != _STORAGE_VERSION:
            return None
        payload = document.get("payload")
        if not isinstance(payload, dict):
            return None
        return payload
    except Exception:
        return None


def purge(
    session_id: str,
    directory: Path | str = DEFAULT_SESSIONS_DIR,
) -> bool:
    """Remove a saved session and report whether a file was removed.

    This function never raises. Missing, invalid, and unreadable paths are
    reported as ``False`` so cleanup cannot interrupt the app.
    """
    try:
        if not is_valid_session_id(session_id):
            return False
        path = Path(directory) / f"{session_id}.json"
        path.unlink()
        return True
    except FileNotFoundError:
        return False
    except Exception:
        return False


def _require_valid_session_id(session_id: object) -> None:
    if not is_valid_session_id(session_id):
        raise ValueError(
            "session id must contain only letters, digits, '-' or '_' "
            f"and be at most {MAX_SESSION_ID_LENGTH} characters"
        )


__all__ = [
    "DEFAULT_SESSIONS_DIR",
    "MAX_SESSION_ID_LENGTH",
    "is_valid_session_id",
    "load",
    "new_session_id",
    "purge",
    "save",
]
