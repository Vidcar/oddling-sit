from __future__ import annotations

import pytest

from oddling.watch import main


def test_watch_help_and_kinds() -> None:
    with pytest.raises(SystemExit) as e:
        main(["watch"])
    assert e.value.code != 0


def test_unknown_rejected() -> None:
    with pytest.raises(SystemExit):
        main(["turbo"])


def test_watch_stupid_without_isaac_reports_missing(monkeypatch, tmp_path) -> None:
    import oddling.lab as lab
    import oddling.watch as watch

    monkeypatch.setattr(watch, "BAT", tmp_path / "missing.bat")
    assert main(["watch", "stupid"]) == 2


def test_watch_trained_without_checkpoint(monkeypatch, tmp_path) -> None:
    import oddling.watch as watch

    monkeypatch.setattr(watch, "trained_checkpoint", lambda: None)
    monkeypatch.setattr(watch, "BAT", tmp_path / "isaaclab.bat")
    (tmp_path / "isaaclab.bat").write_text("echo", encoding="utf-8")
    assert main(["watch", "trained"]) == 2
