from __future__ import annotations

from dataclasses import dataclass


NAMES = ("survive", "move", "upright")


@dataclass(frozen=True)
class Lesson:
    survive: float = 1.0
    move: float = 0.0
    upright: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {"survive": self.survive, "move": self.move, "upright": self.upright}

    @classmethod
    def default(cls) -> Lesson:
        return cls()

    @classmethod
    def from_dict(cls, raw: dict) -> Lesson:
        return cls(
            survive=float(raw.get("survive", 1.0)),
            move=float(raw.get("move", 0.0)),
            upright=float(raw.get("upright", 0.0)),
        )

    def scaled(self, name: str, value: float) -> Lesson:
        d = self.as_dict()
        d[name] = max(0.0, min(1.0, float(value)))
        return Lesson.from_dict(d)
