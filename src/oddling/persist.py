from __future__ import annotations

import json
from pathlib import Path

from flax.serialization import from_bytes, to_bytes

from oddling.lessons import Lesson

USER = Path.home() / ".oddling"
SAVE = USER / "current.json"
PARAMS = USER / "policy.msg"


def save(lesson: Lesson, chart: list[float], steps: int, params) -> None:
    USER.mkdir(parents=True, exist_ok=True)
    SAVE.write_text(
        json.dumps({"lesson": lesson.as_dict(), "chart": chart[-400:], "steps": int(steps)}),
        encoding="utf-8",
    )
    PARAMS.write_bytes(to_bytes(params))


def load_meta() -> dict | None:
    if not SAVE.is_file():
        return None
    return json.loads(SAVE.read_text(encoding="utf-8"))


def load_params(template):
    if not PARAMS.is_file():
        return None
    return from_bytes(template, PARAMS.read_bytes())
