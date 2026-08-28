# Oddling PRD r3 — body can live

**Revision:** r3
**Status:** Approved
**Date:** 2026-08-28
**Approver:** Product owner

**Change from r2:** The Godot-era sitting is not this product. Turbo, progress chart, lessons, persist-as-a-sitting, drop-food, live “learning already going,” and “latest best” are not brought over. Oddling’s intended game is restated: you assemble creatures; you never drive them; they do what they learned in order to survive. This delivery is only the first proof that a body we make can actually do that living thing.

## 1. Problem and outcome

Oddling is meant to be a game where you make a creature and then watch it live. You do not control it. It does whatever it has learned to do to survive. Motion is not a “walk forward” assignment. Survival is the point; a walk is allowed to appear, or not.

Nothing in the current Oddling work is finished. The old sitting (speed, chart, lessons, come-back-as-a-save, snap-kit as a small overlay) came from an earlier attempt and does not describe the game. A stock robot that was told to walk also does not describe the game: that only proves a lab can locomote, not that *our* bodies can invent living.

Until a body we make can actually get food and keep living, the creature-builder game is a sketch on top of an unproven creature.

**Outcome of this delivery:** On this machine you can watch one simple four-leg critter, with a mouth, on a flat field, trying to live. You can watch it **stupid** and **trained**. Trained gets food repeatedly without being told to walk. Stupid mostly misses and collapses. That is enough to treat custom bodies as viable. It is not the Oddling game yet.

## 2. Users and context

- **You**, locally. First player and the person who needs to see that this is real. No account. Nothing to send anyone.
- You are checking viability before building Oddling. You are not playing the creature-builder game in this delivery.
- A later player of Oddling still must never need the name of a trainer. They need a creature that lives, or fails in a way they can read.

You watch a thing try to live. You never move its joints.

## 3. In scope

- One **simple four-leg critter with a clear front and mouth**. The current body is a sketch. It may be rebuilt or refined until eating is actually possible.
- One **flat field**. No hills. No obstacles.
- **Eat or die.** Energy always drains. Living is not a decoration.
- One food already in front at the start of a run. Mouth reaches food → eat. Food vanishes, energy goes up, food reappears a short way off on the same field so it has to get the next one. You never place food.
- **Starve** is on screen: energy hits zero, the body **collapses**, then **that same run continues** (same stupid or same trained). Food is in front again. No game-over. The body is not immortal.
- After the body has been trained, you **pick stupid or trained** and watch **one on screen**. Close and come back: both playings are still there to watch. No chart, no lesson, no sitting-save of a play session.
- **Viable** means: in a short watch, trained **gets food repeatedly**. Ugly is allowed (crawl, flop, roll). One lucky eat then starve is not viable — the body is changed and you watch again. A pretty walk is not required.
- You never puppet joints. The body only does what it has learned (trained) or random flailing (stupid).
- A short guide that matches this delivery: what this is, how to watch stupid and trained, what “works” looks like.

## 4. Out of scope

- The Oddling **game**: creature builder, many bodies, catalogue, sharing, accounts, public release.
- Assembling parts during this delivery. You watch one body. Rebuilding that body is our work between watches, not a kit you use while watching.
- Hills, obstacles, search maps, other worlds (water, sky, mountains).
- A **walk-forward** goal, velocity tracking, or any score that tells the body it must walk.
- Old sitting chrome, brought over or renamed: turbo / speed, pause-as-a-feature, progress chart, brain view, named lessons (survive / move / upright), drop food, “learning already going” as the thing you open, “latest best” as a crowd-vs-hero, persist of lesson and chart history.
- Classroom, Train as a button, overnight job as a feature, the Godot garden as the product.
- Predation, other critters as food, plants that evolve, combat.
- Game-over. Immortal bodies. Promising a polished gait.

A walk-forward robot demo is not this product. Existing lab work that already makes stock bodies walk is not rebuilt here and is not the thing you are judging.

## 5. User flows

**Happy path**
The four-leg body exists. Food is on the flat field. You watch **stupid**: it mostly misses food and collapses; the run continues. You watch **trained**: it invents some way to get food more than once (walk, crawl, flop — whatever). You never touch the joints. You close. You come back and can watch stupid or trained again.

**Starve**
No food reached. Energy hits zero. Collapse. Same run continues with food in front. You can keep watching.

**Trained is not viable**
Trained lucks one eat, or never eats, or only twitches. That is a fail. The body is changed. You watch stupid and trained again. Repeat until trained gets food repeatedly, or it is clear this kind of body cannot.

**Ugly living**
Trained gets food again and again but does not look like a walker. That is a pass for this delivery.

## 6. Product rules

- **Oddling** is the intended game: creatures you make, that live by what they learned, that you never drive. That game is not this delivery.
- This delivery is the **body-can-live** proof for that game.
- One field. One body on screen. You pick **stupid** or **trained**.
- Hands off the body. You do not puppet joints. You do not place food. You do not nudge a lesson.
- Energy always drains. Eat or die. Eat is mouth to food. Death is collapse, then the same run continues.
- Survival is the law. Walking is not.
- The default body starts **stupid**. Trained is the same body after it has learned to live.
- Progress is whether trained **gets food repeatedly**. There is no progress chart. You read living from the world: food vanishes; the body collapses.
- Closing does not throw away the two watchable playings (stupid and trained) of this body.
- If the body cannot live, we change the body. The sketch is not sacred.
- Old Oddling sitting words are not the product: sitting, turbo, lessons, latest best, snap-kit, Classroom, Train.

**Terms**

| Term | Meaning |
|---|---|
| Oddling | The intended game (creature you make; it lives by learning). Not this delivery. |
| Body / critter | This delivery’s one four-leg creature with a mouth |
| Field | The one flat world you watch |
| Food | The reason to live; starts in front; returns nearby after an eat |
| Energy | Spent by living; filled by eating; zero means collapse |
| Stupid | The untrained playing of this body |
| Trained | The playing of this body after it has learned to live |
| Collapse | Visible starve; the run then continues |
| Creature builder | Later: you assemble a creature from parts. Not this delivery. |

## 7. Acceptance criteria

1. This delivery does not present the old Oddling sitting as the product (no turbo, progress chart, lessons, drop food, snap-kit, Classroom, Train, or Godot garden).
2. You can watch one four-leg body with a visible mouth on one flat field. Food is visible. Hills and obstacles are not offered.
3. Food starts in front. When the mouth reaches it, the food is gone and energy has gone up. Food then appears a short way off. You cannot place food.
4. Energy drains. Without eats, the on-screen body reaches zero, **collapses**, then the same stupid or trained run continues with food in front. No game-over.
5. You can pick **stupid** or **trained** and watch **one** body on screen. You never move the joints.
6. In a short watch, **stupid** mostly misses food and collapses.
7. In a short watch, **trained** **gets food repeatedly**. Ugly motion counts. A walk is not required. A single lucky eat does not count as pass.
8. Close and open again: stupid and trained of this body are still watchable without starting from nothing.
9. Nothing offers a walk-forward goal, a lesson slider, or a progress chart as the way you judge the body.
10. A short guide describes this delivery only: watch stupid vs trained, eat or die, what pass looks like.

## 8. Open product decisions

None.

## 9. Deferred product decisions

| Deferred | User-visible boundary this delivery |
|---|---|
| Oddling the game (full rewrite) | You watch one proof body. You do not play Oddling. |
| Creature builder (assemble from parts; many shapes) | One four-leg with a mouth. We may rebuild it between watches. You do not assemble. |
| Hills, obstacles, harder worlds | One flat field |
| Telling the body to walk / move / stay upright as a lesson | Survival only: eat or die |
| Live watch of learning, speed control, chart, brain | Train happens; you judge by watching stupid vs trained |
| Drop food, nudge the world while watching | Food is already there and returns nearby |
| Catalogue, sharing, accounts, many critters | One body. Local only. |
| Polished gait in a short watch | Repeated eats; ugly allowed |
| Predation, evolving plants, other worlds | Food on a flat field |

The intended game remains: you make a creature; you never control it; everything that moves does so because it learned how to survive. This delivery only proves that a body we make can.

## 10. Approval record

| | |
|---|---|
| Status | Approved |
| Approver | Product owner |
| Date | 2026-08-28 |
| Revision | Oddling PRD r3 — body can live |
