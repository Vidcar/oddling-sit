from __future__ import annotations

from pathlib import Path

from oddling.env import XML_PATH, Field, load_mj_model
from oddling.lessons import Lesson


def test_xml_exists() -> None:
    assert XML_PATH.is_file()
    model = load_mj_model()
    assert model.nu == 8
    assert model.nq >= 15


def test_lesson_default_and_reset() -> None:
    d = Lesson.default()
    assert d.survive == 1.0
    assert d.move == 0.0
    scaled = d.scaled("move", 0.5)
    assert scaled.move == 0.5
    assert Lesson.from_dict(scaled.as_dict()).move == 0.5


def test_readme_is_sitting_guide() -> None:
    text = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8").lower()
    assert "idad" not in text
    assert "no classroom" in text
    assert "no overnight" in text
    assert "turbo" in text
    assert "chart" in text
    assert "train" not in text or "no train" in text


def test_prd_r2() -> None:
    prd = (Path(__file__).resolve().parents[1] / "docs" / "prd.md").read_text(encoding="utf-8")
    assert "Oddling PRD r2" in prd
    assert "Approved" in prd
    assert "Godot garden is not this product" in prd
