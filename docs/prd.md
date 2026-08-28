# Oddling PRD r2 — GPU-sim sitting (the start)

**Revision:** r2
**Status:** Approved
**Date:** 2026-08-28
**Approver:** Product owner

**Change from r1:** The world is a fast 3D simulation, not the Godot garden. Overnight training is gone. The default body starts stupid. Snap-kit and the Godot overlay pile are out of this sitting. Progress is a chart/number you sit with; a finished gait in a short sit is not promised.

## 1. Problem and outcome

Someone wants to sit down and *see* a 3D body invent how to live — not leave a job overnight, not puppet joints, not read a paper. Game engines that look like a garden cannot run that learning fast enough. A raw simulator is fast but is not a sitting.

**Outcome:** You open Oddling on a local machine with a real GPU. You are in one 3D field with one stupid default body. You turbo, watch the latest best, and read a chart/number over time. It may rise slowly and plateau. That is the game. The Godot garden is not this product.

## 2. Users and context

- **You**, locally, first player. A capable GPU is required. No account. Nothing to send anyone.
- Built so a curious person could play later: they do not need the name of the learner. They need the body, the field, turbo, and the chart.
- Sharing is not this delivery. A weak GPU is not a broken game; the chart still moves, slower.

You sit to watch a thing invent how to live, or fail in a way you can read.

## 3. In scope

- One 3D field. Land straight in. Default body already there, food already there, learning already going in the sitting.
- Default body starts **stupid**.
- Watch the latest best on screen. Other copies are not a crowd.
- Pause, realtime, and **turbo**.
- A **progress chart/number over time**. Slow rise and plateau are allowed and visible. A finished walk in a short sit is not required.
- Brain view so you can see why it is or is not coming along.
- Named lessons you can leave or nudge: **survive**, **move**, **upright**. A way to return the lesson to default.
- Energy always drains. Eat or die. Drop food.
- Death is on screen (collapse). Then the latest best of this critter returns. Sitting does not end.
- Quit and return: this critter, latest best, lesson, and chart history are still there.
- You never move the joints.
- One current critter. This delivery is **one default body**. That body owes a chart that can move while you sit.
- No Train, no overnight job, no Classroom, no Godot garden overlay pile as the sitting.

## 4. Out of scope

- The Godot Oddling repository as the gym or the sitting.
- Overnight / Train as a feature.
- Snap-kit, Spore editor, paint, unlimited parts.
- Other worlds (water, sky, mountains, thin air).
- Plants that evolve; predation; other critters as food.
- Catalogue / bestiary of many bodies.
- Sharing, accounts, public game.
- Height / fly / jump lessons; antennae.
- Custom reward language / joke laws.
- Game-over. Immortal bodies.
- Promising a polished 3D gait in a few minutes.

## 5. User flows

**Happy path**
Open Oddling (GPU machine). Field, stupid default body, food, learning already going. Watch the latest best. Open chart and brain. Turbo. The number moves, or plateaus in a way you can read. Nudge a lesson or drop food, or leave it. Death: collapse, then latest best returns. Quit. Come back: same critter, latest best, lesson, chart.

**Leave it**
Never touch lessons. Default survive. Only watch, turbo, maybe drop food.

**Starve**
No food or cannot eat. Energy hits zero. Collapse. Latest best returns. Drop food. Sitting continues.

**Hard body / slow learning**
The number moves little and plateaus. That is not a broken game. Chart and brain still show what it is doing. You can change the lesson, turbo, or keep watching.

## 6. Product rules

- **Oddling** is the name of this sitting.
- The world that learns is the world you watch. Not a pretty shell over a different gym.
- One field. One current critter. On-screen body is the latest best.
- Hands: world and lesson only. Never puppet joints.
- Energy always drains. Eat or die.
- Default lesson: survive (find food, eat, don’t starve). Move and upright can be nudged. Leave-it is valid.
- Default body starts stupid. No handed-over finished gait.
- Progress is the chart/number over time. Plateau is honest.
- A capable local GPU is required. Slower machines still show the chart; they are not a second product.
- Closing the game does not throw away this critter’s body, latest best, lesson, or chart history.
- Copies that are not the latest best are not something you have to watch as a crowd.
- Overnight Train and the Godot garden sitting are not this product.

## 7. Acceptance criteria

1. Opening this sitting does not present the Godot garden, a Train/overnight job, or a Classroom room.
2. Open lands in one 3D field: default body visible, food visible, learning already going. Body is clearly not a finished walker.
3. On-screen body is the latest best, not a crowd of ragdolls.
4. Pause, realtime, and turbo all work without moving joints.
5. A progress chart or number over time is visible and updates while you sit. A plateau is visible as a plateau, not as a frozen or hidden meter.
6. Brain is available and readable against survive / move / upright.
7. Survive, move, and upright can be left at default or nudged; lesson can be returned to default.
8. Energy drains; without food the on-screen body reaches death and collapses; latest best returns; sitting continues.
9. Drop food works.
10. Quit and reopen: same current critter, latest best, lesson, and chart history.
11. Snap-kit, other worlds, catalogue, sharing, and Train are not offered.
12. On a capable GPU, you can sit and see the chart move without leaving an overnight job. A finished gait in a short sit is not required.

## 8. Open product decisions

None.

## 9. Deferred product decisions

| Deferred | User-visible boundary this delivery |
|---|---|
| Snap-kit / Spore editor | One default body only |
| Other worlds; evolving plants; predation | One field; food sits there; you can drop more |
| Catalogue / share / accounts | One current critter; local only |
| Overnight Train as a tool | Not offered |
| Godot garden as a skin | Not this product |
| Finished gait in minutes | Chart/number is the progress; gait may lag or plateau |

## 10. Approval record

| | |
|---|---|
| Status | Approved |
| Approver | Product owner |
| Date | 2026-08-28 |
| Revision | Oddling PRD r2 — GPU-sim sitting (the start) |
