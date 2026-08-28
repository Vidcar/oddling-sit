# Oddling

Watch a four-leg critter try to live. You never move its joints.

This is the body-can-live check. Not the Oddling game.

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
- Food starts in front. Mouth reaches it → eat. Food comes back nearby. Energy always drains. Eat or die.
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

Local Windows. One body. Flat field. Eat or die.
