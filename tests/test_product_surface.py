from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_prd_r3() -> None:
    prd = (ROOT / "docs" / "prd.md").read_text(encoding="utf-8")
    assert "Oddling PRD r3" in prd
    assert "Approved" in prd
    assert "walk-forward" in prd.lower() or "walk forward" in prd.lower()


def test_readme_is_watch_guide() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8").lower()
    assert "watch stupid" in text
    assert "watch trained" in text
    assert "eat or die" in text
    assert "turbo" not in text
    assert "classroom" not in text
    assert "lesson" not in text
