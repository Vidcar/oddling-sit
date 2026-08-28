from pathlib import Path


def test_prd_r3_not_r2() -> None:
    prd = (Path(__file__).resolve().parents[1] / "docs" / "prd.md").read_text(encoding="utf-8")
    assert "Oddling PRD r3" in prd
    assert "Draft - awaiting approval" not in prd
