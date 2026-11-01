"""Tests for opaque session identifiers and JSON session persistence."""

from __future__ import annotations

from pathlib import Path

import pytest

from heartrisk.session import (
    is_valid_session_id,
    load,
    new_session_id,
    purge,
    save,
)


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    session_id = new_session_id()
    payload = {
        "answers": {"age": 62},
        "skipped": ["sex"],
        "pending": [],
        "finished": False,
    }

    save(session_id, payload, directory=tmp_path)

    assert load(session_id, directory=tmp_path) == payload


def test_load_of_an_unknown_session_is_none(tmp_path: Path) -> None:
    assert load(new_session_id(), directory=tmp_path) is None


def test_load_of_a_corrupt_file_is_none(tmp_path: Path) -> None:
    session_id = new_session_id()
    (tmp_path / f"{session_id}.json").write_text("{not valid json", encoding="utf-8")

    assert load(session_id, directory=tmp_path) is None


def test_load_of_an_unreadable_file_is_none(tmp_path: Path) -> None:
    session_id = new_session_id()
    (tmp_path / f"{session_id}.json").mkdir()

    assert load(session_id, directory=tmp_path) is None


def test_load_of_an_older_file_format_is_none(tmp_path: Path) -> None:
    session_id = new_session_id()
    (tmp_path / f"{session_id}.json").write_text(
        '{"version": 0, "payload": {"answers": {}}}', encoding="utf-8"
    )

    assert load(session_id, directory=tmp_path) is None


def test_save_rejects_a_non_serializable_payload(tmp_path: Path) -> None:
    session_id = new_session_id()

    with pytest.raises(TypeError, match="JSON-serial"):
        save(session_id, {"answers": object()}, directory=tmp_path)

    assert load(session_id, directory=tmp_path) is None


@pytest.mark.parametrize(
    "value",
    ["../x", "/etc/passwd", "", "a" * 10_000, "session.json", "with/slash"],
)
def test_invalid_session_ids_are_rejected(value: str) -> None:
    assert is_valid_session_id(value) is False


def test_a_generated_session_id_is_valid() -> None:
    assert is_valid_session_id(new_session_id()) is True


def test_save_rejects_an_invalid_session_id(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="session id"):
        save("../x", {"answers": {}}, directory=tmp_path)


def test_save_leaves_no_temporary_files_behind(tmp_path: Path) -> None:
    session_id = new_session_id()

    save(session_id, {"answers": {}}, directory=tmp_path)

    assert [path.name for path in tmp_path.iterdir()] == [f"{session_id}.json"]


def test_a_failed_atomic_replace_keeps_the_previous_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session_id = new_session_id()
    first_payload = {"answers": {"age": 62}}
    save(session_id, first_payload, directory=tmp_path)

    def fail_replace(source: object, destination: object) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr("heartrisk.session.os.replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        save(session_id, {"answers": {"age": 63}}, directory=tmp_path)

    assert load(session_id, directory=tmp_path) == first_payload
    assert [path.name for path in tmp_path.iterdir()] == [f"{session_id}.json"]


def test_purge_removes_an_existing_session(tmp_path: Path) -> None:
    session_id = new_session_id()
    save(session_id, {"answers": {}}, directory=tmp_path)

    assert purge(session_id, directory=tmp_path) is True
    assert load(session_id, directory=tmp_path) is None
    assert purge(session_id, directory=tmp_path) is False


def test_purge_never_raises_for_an_unreadable_path(tmp_path: Path) -> None:
    session_id = new_session_id()
    (tmp_path / f"{session_id}.json").mkdir()

    assert purge(session_id, directory=tmp_path) is False
