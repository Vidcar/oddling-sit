from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

XML = Path(__file__).resolve().parents[1] / "src" / "oddling" / "body.xml"


def test_four_leg_mouth() -> None:
    root = ET.parse(XML).getroot()
    names = {b.get("name") for b in root.iter("body")}
    assert "torso" in names
    for leg in ("fl_hip", "fr_hip", "hl_hip", "hr_hip", "fl_shin", "fr_shin", "hl_shin", "hr_shin"):
        assert leg in names
    assert "mouth" in names
    motors = root.findall("./actuator/motor")
    assert len(motors) == 8
    geoms = {g.get("name") for g in root.iter("geom")}
    assert "food" not in geoms
    assert "floor" not in geoms
    assert root.find("./worldbody/body[@name='food']") is None
