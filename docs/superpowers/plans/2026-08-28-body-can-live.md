# Body Can Live Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Tasks are tightly coupled (body → live rules → Isaac Lab env → train → watch). Do not use subagent-driven-development in parallel.

**Goal:** PRD r3 — a four-leg critter with a mouth on a flat field; watch stupid vs trained; trained gets food repeatedly without a walk-forward score.

**Architecture:** Keep living rules (energy, eat, food respawn, collapse-then-continue) as a pure Python module tested without the GPU lab. The GPU lab is Isaac Lab at `D:\codeprojects\IsaacLab` (already installed, rsl_rl already trained a stock ant). An external Isaac Lab extension in this repo registers `Oddling-Live-Direct`. Training and playback use Isaac Lab’s existing rsl_rl `train.py` / `play.py`. Do not rebuild PPO. Do not use Ant’s walk-forward target. Stupid = untrained policy. Trained = saved checkpoint.

**Tech Stack:** Python 3.12, Isaac Lab 3 (Isaac Sim 6) at `D:\codeprojects\IsaacLab`, `env_isaaclab` venv, rsl_rl via `isaaclab_rl`, gymnasium, torch, MJCF body spawned with `MjcfFileCfg`, pytest for living-rules and product-surface tests.

## Global Constraints

- Product is `docs/prd.md` r3 approved 2026-08-28. Do not reintroduce turbo, progress chart, lessons, drop-food, snap-kit, Classroom, Train-as-a-button, Godot garden, or walk-forward scoring.
- One four-leg body with a visible mouth. Body may be rebuilt if eating cannot happen.
- Flat field. Energy always drains. Mouth-to-food is eat. Food starts in front, reappears nearby after eat. Collapse then same run continues.
- Watch: pick stupid or trained, one on screen. Close and return: both still launchable.
- Viable: trained gets food repeatedly in a short watch; stupid mostly misses and collapses; ugly allowed.
- You never puppet joints.
- Local Windows. GPU lab path: `D:\codeprojects\IsaacLab`. Python: `D:\codeprojects\IsaacLab\env_isaaclab\Scripts\python.exe`. Launcher: `D:\codeprojects\IsaacLab\isaaclab.bat -p`.
- Verify (fresh): `D:\codeprojects\oddling-sit\.worktrees\body-can-live\.venv` is NOT the Isaac venv. Living-rules tests: repo pytest. Lab: Isaac python. Product: `oddling watch stupid` and `oddling watch trained` plus a headless eat-count eval.
- Work on branch `feat/body-can-live` in this worktree. Commit each verified task. Do not merge to the main line.

## File map

- `docs/prd.md` — already r3 approved
- `README.md` — watch guide only
- `src/oddling/live.py` — pure living rules
- `src/oddling/body.xml` — four-leg + mouth MJCF
- `src/oddling/watch.py` — CLI: watch stupid | watch trained | train
- `src/oddling/lab.py` — Isaac Lab path, task id, checkpoint dir
- `source/oddling_lab/` — Isaac Lab extension (gym env)
- `tests/test_live.py`, `tests/test_body.py`, `tests/test_product_surface.py`, `tests/test_watch_cli.py`
- `runs/trained/` — gitignored; trained checkpoint copy so quit/reopen works

---

### Task 1: Product surface + living rules

**Files:**
- Create: `src/oddling/live.py`
- Create: `tests/test_live.py`
- Create: `tests/test_product_surface.py`
- Modify: `README.md`
- Modify: `tests/test_field.py` (replace r2 sitting assertions)

**Interfaces:**
- Consumes: PRD r3
- Produces: `LiveState`, `step_live(state, mouth_xy, dt_steps=1) -> LiveState`, constants `START_ENERGY`, `DRAIN`, `EAT_RADIUS`, `EAT_ENERGY`, `FOOD_HOME`, `FOOD_NEAR_XY`

- [ ] **Step 1: Write failing living-rules tests**

```python
from oddling.live import (
    DRAIN,
    EAT_ENERGY,
    EAT_RADIUS,
    FOOD_HOME,
    START_ENERGY,
    LiveState,
    step_live,
)


def test_energy_drains_without_eat():
    st = LiveState.spawn()
    nxt = step_live(st, mouth_xyz=FOOD_HOME, ate=False)
    assert nxt.energy == START_ENERGY - DRAIN
    assert nxt.eats == 0
    assert nxt.collapsed is False


def test_mouth_on_food_eats_and_food_moves():
    st = LiveState.spawn()
    nxt = step_live(st, mouth_xyz=FOOD_HOME, ate=True)
    assert nxt.energy == START_ENERGY - DRAIN + EAT_ENERGY
    assert nxt.eats == 1
    assert nxt.food != FOOD_HOME
    dx = abs(nxt.food[0] - FOOD_HOME[0])
    dy = abs(nxt.food[1] - FOOD_HOME[1])
    assert dx <= 0.8 + 1e-5
    assert dy <= 0.5 + 1e-5


def test_starve_collapses_then_same_run_continues():
    st = LiveState.spawn()
    st = LiveState(energy=DRAIN * 0.5, eats=0, food=FOOD_HOME, collapsed=False, collapse_left=0)
    dead = step_live(st, mouth_xyz=(0.0, 0.0, 0.4), ate=False)
    assert dead.collapsed is True
    assert dead.collapse_left > 0
    while dead.collapse_left > 0:
        dead = step_live(dead, mouth_xyz=(0.0, 0.0, 0.05), ate=False)
    assert dead.collapsed is False
    assert dead.energy == START_ENERGY
    assert dead.food == FOOD_HOME
```

- [ ] **Step 2: Run tests, expect fail**

Run: `.\.venv\Scripts\python -m pytest tests/test_live.py -q`
Expected: FAIL (module missing) from worktree; if no local venv, `python -m pytest` with `PYTHONPATH=src`.

- [ ] **Step 3: Implement `src/oddling/live.py`**

```python
from __future__ import annotations

from dataclasses import dataclass

START_ENERGY = 12.0
DRAIN = 0.04
EAT_RADIUS = 0.45
EAT_ENERGY = 4.0
FOOD_HOME = (2.2, 0.0, 0.12)
FOOD_JITTER_X = 0.8
FOOD_JITTER_Y = 0.5
COLLAPSE_STEPS = 24


def mouth_reaches(mouth_xyz: tuple[float, float, float], food_xyz: tuple[float, float, float]) -> bool:
    dx = mouth_xyz[0] - food_xyz[0]
    dy = mouth_xyz[1] - food_xyz[1]
    dz = mouth_xyz[2] - food_xyz[2]
    return (dx * dx + dy * dy + dz * dz) ** 0.5 < EAT_RADIUS


@dataclass(frozen=True)
class LiveState:
    energy: float
    eats: int
    food: tuple[float, float, float]
    collapsed: bool
    collapse_left: int
    rng: int = 1

    @classmethod
    def spawn(cls) -> LiveState:
        return cls(energy=START_ENERGY, eats=0, food=FOOD_HOME, collapsed=False, collapse_left=0, rng=1)


def _jitter(rng: int) -> tuple[float, float, float]:
    # deterministic LCG so tests are stable
    rng = (1103515245 * rng + 12345) & 0x7FFFFFFF
    ux = (rng % 1000) / 1000.0 * FOOD_JITTER_X
    rng = (1103515245 * rng + 12345) & 0x7FFFFFFF
    uy = ((rng % 1000) / 1000.0 * 2.0 - 1.0) * FOOD_JITTER_Y
    return (FOOD_HOME[0] + ux, FOOD_HOME[1] + uy, FOOD_HOME[2]), rng


def step_live(st: LiveState, mouth_xyz: tuple[float, float, float], ate: bool | None = None) -> LiveState:
    if ate is None:
        ate = (not st.collapsed) and mouth_reaches(mouth_xyz, st.food)
    if st.collapsed:
        left = st.collapse_left - 1
        if left <= 0:
            return LiveState.spawn().__class__(
                energy=START_ENERGY,
                eats=st.eats,
                food=FOOD_HOME,
                collapsed=False,
                collapse_left=0,
                rng=st.rng,
            )
        return LiveState(st.energy, st.eats, st.food, True, left, st.rng)
    energy = st.energy - DRAIN + (EAT_ENERGY if ate else 0.0)
    eats = st.eats + (1 if ate else 0)
    food, rng = st.food, st.rng
    if ate:
        food, rng = _jitter(st.rng)
    if energy <= 0.0:
        return LiveState(0.0, eats, food, True, COLLAPSE_STEPS, rng)
    return LiveState(energy, eats, food, False, 0, rng)
```

Fix `_jitter` to return a tuple of food and rng (the sketch above mixed a nested tuple). Implement as:

```python
def _jitter(rng: int) -> tuple[tuple[float, float, float], int]:
    rng = (1103515245 * rng + 12345) & 0x7FFFFFFF
    ux = (rng % 1000) / 1000.0 * FOOD_JITTER_X
    rng = (1103515245 * rng + 12345) & 0x7FFFFFFF
    uy = ((rng % 1000) / 1000.0 * 2.0 - 1.0) * FOOD_JITTER_Y
    return (FOOD_HOME[0] + ux, FOOD_HOME[1] + uy, FOOD_HOME[2]), rng
```

And spawn-after-collapse keeps `eats` (same run continues; eats is cumulative for this watch) OR resets eats. PRD: “that same run continues” with food in front — reset energy and food, keep it as the same stupid/trained playing. Reset eats to 0 on collapse-complete so each life is readable. Tests should expect eats reset on continue.

Update test_starve: after collapse finishes, `eats == 0`, `food == FOOD_HOME`, `energy == START_ENERGY`.

- [ ] **Step 4: Product-surface tests + README**

`tests/test_product_surface.py`:

```python
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_prd_r3():
    prd = (ROOT / "docs" / "prd.md").read_text(encoding="utf-8")
    assert "Oddling PRD r3" in prd
    assert "Approved" in prd
    assert "walk-forward" in prd.lower() or "walk forward" in prd.lower()


def test_readme_is_watch_guide():
    text = (ROOT / "README.md").read_text(encoding="utf-8").lower()
    assert "watch stupid" in text
    assert "watch trained" in text
    assert "turbo" not in text
    assert "classroom" not in text
    assert "lesson" not in text
    assert "no overnight" in text or "not a sitting" in text or "eat or die" in text
```

Rewrite `README.md`:

```markdown
# Oddling

Watch a four-leg critter try to live. You never move its joints.

This delivery is the body-can-live check. Not the Oddling game.

## Watch

Isaac Lab at `D:\codeprojects\IsaacLab` is required (its `env_isaaclab` venv).

```powershell
cd D:\codeprojects\oddling-sit
D:\codeprojects\IsaacLab\env_isaaclab\Scripts\pip install -e ".[dev]"
oddling watch stupid
oddling watch trained
```

- `watch stupid` — untrained body. It mostly misses food and collapses. The run continues.
- `watch trained` — same body after it learned to live. It should get food more than once. Ugly is allowed.
- Food starts in front. Mouth reaches it → eat. Food comes back nearby. Energy always drains.
- Close and come back: both watchings are still there after a train.

```powershell
oddling train
```

That takes a while. Then `watch trained` uses the saved body.

## Check

```powershell
.\.venv\Scripts\python -m pytest -q
```

## Envelope

Local Windows. One body. Flat field. Eat or die. No turbo, no chart, no lessons.
```

Fix README so `test_readme_is_watch_guide` matches actual wording. Include `eat or die`. Do not mention turbo except to say it is not offered — wait, the test says `turbo not in text`. So do not mention turbo at all.

- [ ] **Step 5: pytest green for Task 1**

Run: `python -m pytest tests/test_live.py tests/test_product_surface.py -q` with `PYTHONPATH=src` (worktree may use `D:\codeprojects\oddling-sit\.venv` if present, or Isaac python).

Expected: PASS

- [ ] **Step 6: Commit**

```
git add src/oddling/live.py tests/test_live.py tests/test_product_surface.py README.md tests/test_field.py
git commit -m "feat: living rules and r3 watch guide"
```

If `tests/test_field.py` still asserts r2, update it so it does not fail: drop r2-only strings; keep xml exists if body not yet changed, or move xml asserts to Task 2.

---

### Task 2: Four-leg body with mouth

**Files:**
- Modify: `src/oddling/body.xml`
- Create: `tests/test_body.py`

**Interfaces:**
- Produces: MJCF with torso, four legs (hip+knee each), site `mouth`, geom `food` removed from the world (food is a separate lab object). `nu == 8`. Named mouth site.

- [ ] **Step 1: Failing test**

```python
from oddling.env import XML_PATH, load_mj_model
```

Do **not** depend on old `oddling.env` MJX Field. Load XML with stdlib:

```python
import xml.etree.ElementTree as ET
from pathlib import Path

XML = Path(__file__).resolve().parents[1] / "src" / "oddling" / "body.xml"


def test_four_leg_mouth():
    root = ET.parse(XML).getroot()
    names = {b.get("name") for b in root.iter("body")}
    assert "torso" in names
    for leg in ("fl_hip", "fr_hip", "hl_hip", "hr_hip", "fl_shin", "fr_shin", "hl_shin", "hr_shin"):
        assert leg in names
    sites = {s.get("name") for s in root.iter("site")}
    assert "mouth" in sites
    motors = list(root.iter("motor"))
    assert len(motors) == 8
    # food is not part of the body asset
    geoms = {g.get("name") for g in root.iter("geom")}
    assert "food" not in geoms
```

- [ ] **Step 2: Run, expect fail** (food geom still in current xml)

- [ ] **Step 3: Edit `body.xml`** — keep the four-leg critter, keep mouth site on torso front, **delete** the mocap food body. Keep freejoint torso. Mouth site `pos="0.22 0 0"` size 0.03, red.

- [ ] **Step 4: pytest tests/test_body.py PASS**

- [ ] **Step 5: Commit** `feat: four-leg body with mouth, food is not in the asset`

---

### Task 3: Isaac Lab extension + eat-or-die env

**Files:**
- Create: `source/oddling_lab/pyproject.toml`
- Create: `source/oddling_lab/oddling_lab/__init__.py`
- Create: `source/oddling_lab/oddling_lab/tasks/__init__.py` (imports live task so gym.register runs)
- Create: `source/oddling_lab/oddling_lab/tasks/live/__init__.py`
- Create: `source/oddling_lab/oddling_lab/tasks/live/live_env_cfg.py`
- Create: `source/oddling_lab/oddling_lab/tasks/live/live_env.py`
- Create: `source/oddling_lab/oddling_lab/tasks/live/agents/__init__.py`
- Create: `source/oddling_lab/oddling_lab/tasks/live/agents/rsl_rl_ppo_cfg.py`
- Create: `source/oddling_lab/oddling_lab/assets/critter.py`
- Create: `src/oddling/lab.py`

**Interfaces:**
- Gym id: `Oddling-Live-Direct`
- Entry: `oddling_lab.tasks.live.live_env:LiveEnv`
- rsl_rl cfg: `LivePPORunnerCfg` experiment_name=`oddling_live`
- Rewards: eat, stay alive, get closer to food. **No** lin_vel_x target, **no** heading-to-1000m.
- Food: RigidObject sphere, mocap or kinematic, repositioned on eat.
- Collapse: energy<=0 → zero actions for COLLAPSE_STEPS then reset (same policy).

Pattern: copy scene setup from `D:\codeprojects\IsaacLab\source\isaaclab_tasks\isaaclab_tasks\core\cartpole\cartpole_direct_env.py` (`_setup_scene`, DirectRLEnv). Spawn robot via `MjcfFileCfg(asset_path=<body.xml>)`. Spawn food via `sim_utils.SphereCfg` + `RigidObjectCfg` with kinematic=True (or high mass disable gravity + pose writes).

`critter.py`:

```python
from pathlib import Path
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

XML = Path(__file__).resolve().parents[4] / "src" / "oddling" / "body.xml"
# parents: assets/ -> oddling_lab/ -> source/oddling_lab/ -> repo root is parents[3]?
# file: source/oddling_lab/oddling_lab/assets/critter.py
# parents[0]=assets, [1]=oddling_lab pkg, [2]=source/oddling_lab, [3]=source, [4]=repo
# Use a stable helper in oddling.lab instead.

CRITTER_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.MjcfFileCfg(
        asset_path=str(XML),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, max_depenetration_velocity=10.0),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.42)),
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=0.0,
            damping=0.1,
            effort_limit=8.0,
        ),
    },
)
```

Resolve XML via `oddling` package:

```python
import oddling
from pathlib import Path
XML = Path(oddling.__file__).with_name("body.xml")
```

Install the extension into Isaac’s venv:

```
D:\codeprojects\IsaacLab\env_isaaclab\Scripts\pip install -e D:\codeprojects\oddling-sit\.worktrees\body-can-live
D:\codeprojects\IsaacLab\env_isaaclab\Scripts\pip install -e D:\codeprojects\oddling-sit\.worktrees\body-can-live\source\oddling_lab
```

`pyproject.toml` (repo) already has package oddling. Add nothing Isaac-specific as required deps (Isaac stays in its venv).

`source/oddling_lab/pyproject.toml`:

```toml
[project]
name = "oddling_lab"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = []

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_backend"

[tool.setuptools.packages.find]
where = ["."]
```

Gym register in `tasks/live/__init__.py` like Ant, id `Oddling-Live-Direct`, default_agent rsl_rl.

**LiveEnv rewards (torch, batched):**

```
dist = ||mouth - food||
ate = dist < EAT_RADIUS
rew = 3.0 * ate.float() + 0.02 * alive.float() - 0.1 * dist.clamp(max=4.0) / 4.0
# NO velocity-x, NO heading
```

Energy/eats/food/collapse tensors on GPU, logic matching `oddling.live` (same constants).

Observations: joint pos, joint vel, torso up, food relative, energy.

Play num_envs=1 for watch. Train num_envs=4096 (or 1024 if VRAM tight; 3090 is available).

- [ ] **Step 1: Write a headless smoke that does not need a trained policy**

`tests/test_lab_smoke.py` should skip if Isaac python is not the current interpreter:

```python
import os
import shutil
import pytest

pytestmark = pytest.mark.skipif(
    shutil.which("isaaclab") is None and "isaaclab" not in os.environ.get("VIRTUAL_ENV", "").lower(),
    reason="Isaac Lab venv only",
)
```

Do not run this under repo `.venv` pytest. Instead Task 3 verification is:

```
D:\codeprojects\IsaacLab\isaaclab.bat -p -c "import oddling_lab.tasks.live; import gymnasium as gym; print(gym.spec('Oddling-Live-Direct'))"
```

Expected: prints spec.

A small script `scripts/eval_eats.py` (Task 5) counts eats.

- [ ] **Step 2: Implement extension files** (DirectRLEnv, cfg, register, rsl_rl runner). Follow cartpole `_setup_scene` clone/replicate. Add food RigidObject. Find mouth body/site after spawn (`find_bodies` / `find_links` — inspect Articulation API if `find_bodies` missing; fallback: index of link named `torso` plus mouth offset in torso frame).

If MJCF site is not imported, add a dummy mouth link in XML (`body name="mouth"` with a small sphere, mass near 0, welded) so it has a body id.

- [ ] **Step 3: Install editable and list the gym id** (command above). Expected: Oddling-Live-Direct registered.

- [ ] **Step 4: Commit** `feat: Oddling-Live-Direct eat-or-die env on Isaac Lab`

---

### Task 4: Watch / train CLI

**Files:**
- Create: `src/oddling/lab.py`
- Modify: `src/oddling/watch.py` (replace pygame sit)
- Modify: `src/oddling/sit.py` — `main` should call watch CLI or print the guide; do not launch pygame sitting
- Modify: `pyproject.toml` scripts: `oddling = "oddling.watch:main"`
- Create: `tests/test_watch_cli.py`

**Interfaces:**
- `oddling watch stupid` → Isaac `play.py` / `random_agent.py` with `--task Oddling-Live-Direct` and **no checkpoint** (or `--use_last_checkpoint false` plus freshly init; prefer Isaac `scripts/environments/random_agent.py` so it is visibly stupid).
- `oddling watch trained` → Isaac `scripts/reinforcement_learning/play.py --task Oddling-Live-Direct --checkpoint <runs/trained/latest>`
- `oddling train` → Isaac `scripts/reinforcement_learning/train.py --task Oddling-Live-Direct --headless` then copy last checkpoint to `runs/trained/`
- Close/reopen: `runs/trained/` plus rsl_rl logs under repo `logs/rsl_rl/oddling_live/` (gitignored)

`lab.py`:

```python
from pathlib import Path
import os

ISAACLAB = Path(os.environ.get("ISAACLAB_PATH", r"D:\codeprojects\IsaacLab"))
TASK = "Oddling-Live-Direct"
PYTHON = ISAACLAB / "env_isaaclab" / "Scripts" / "python.exe"
BAT = ISAACLAB / "isaaclab.bat"
TRAINED = Path.home() / ".oddling" / "trained"
# also copy to repo runs/trained for the worktree
```

Use `~/.oddling/trained` so quit/reopen survives worktree paths. Create that dir on train.

`tests/test_watch_cli.py` (no GPU): argparse accepts `watch stupid`, `watch trained`, `train`; rejects `turbo`.

- [ ] **Step 1: Failing CLI test**
- [ ] **Step 2: Implement watch.py**
- [ ] **Step 3: pytest tests/test_watch_cli.py PASS**
- [ ] **Step 4: Commit** `feat: oddling watch stupid|trained and train CLI`

---

### Task 5: Train and prove trained gets food

**Files:**
- Create: `scripts/eval_eats.py` — headless, N seconds, print `eats=<int>`
- Train for real on the 3090
- Copy checkpoint to `~/.oddling/trained`

**Eval:** run policy in `Oddling-Live-Direct` with `num_envs=1` (or 16) for ~30s sim. Count eats.

Pass bar (PRD):
- stupid (random): eats == 0 or very rare; collapse happens
- trained: eats >= 3 in a short watch (repeated)

If trained fails: change body (joint ranges, mouth, friction, effort) and/or reward scales (still no walk-forward), retrain. Do not add lin_vel_x.

Train command:

```
D:\codeprojects\IsaacLab\isaaclab.bat -p D:\codeprojects\IsaacLab\scripts\reinforcement_learning\train.py --task Oddling-Live-Direct --headless --max_iterations 1000
```

Ant already ran 1000 iterations on this machine; budget similarly.

- [ ] **Step 1: `eval_eats.py` works on random policy (low eats)**
- [ ] **Step 2: Train to 1000 iterations (or until eval_eats trained >= 3)**
- [ ] **Step 3: Copy checkpoint; `oddling watch trained` launches**
- [ ] **Step 4: Record eval numbers in the commit message / a `runs/EVAL.txt` gitignored; put the numbers in the delivery report. Commit code only.**
- [ ] **Step 5: Commit** `feat: train live body and persist trained checkpoint path`

---

### Task 6: Fresh verification vs PRD r3

Re-read `docs/prd.md` acceptance 1–10. Run:

```
python -m pytest tests -q
oddling watch stupid   # smoke: process starts, field+body+food (manual / screenshot if GUI)
oddling watch trained
```

Headless:

```
isaaclab.bat -p scripts/eval_eats.py --policy stupid
isaaclab.bat -p scripts/eval_eats.py --policy trained
```

Checklist:

1. No turbo/chart/lessons/drop-food/snap-kit/Classroom/Train/Godot in README or CLI
2. Four-leg + mouth, flat field, food visible
3. Food in front, eat, nearby respawn
4. Collapse then continue
5. Pick stupid or trained, no joint puppet
6. Stupid mostly misses
7. Trained repeated eats
8. Quit/reopen trained still there
9. No walk-forward goal
10. README matches this delivery

- [ ] **Step 1: Fresh pytest**
- [ ] **Step 2: Fresh eval_eats stupid vs trained**
- [ ] **Step 3: Push branch `feat/body-can-live`**
- [ ] **Step 4: Stop. Show worktree path and evidence. Do not merge.**

---

## Spec coverage

| PRD acceptance | Task |
|---|---|
| 1 no old sitting | 1, 4 |
| 2 four-leg mouth flat field food | 2, 3 |
| 3 eat and nearby food | 1, 3 |
| 4 collapse then continue | 1, 3 |
| 5 pick stupid or trained | 4 |
| 6 stupid misses | 5 |
| 7 trained repeated eats | 5 |
| 8 quit/reopen | 4, 5 |
| 9 no walk-forward / chart / lessons | 3, 4 |
| 10 short guide | 1 |

## Existing work (do not rebuild)

- Isaac Lab DirectRLEnv, cloner, MjcfFileCfg, rsl_rl train/play
- Stock Ant walk task is **not** the product; do not register Oddling as Isaac-Ant
- Old pygame MJX PPO sitting in `sit.py` / `trainer.py` is not the product; CLI must not launch it
