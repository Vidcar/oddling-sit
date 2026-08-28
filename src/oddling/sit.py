from __future__ import annotations

import sys

import mujoco
import numpy as np
import pygame

from oddling.env import FOOD_HOME, load_mj_model
from oddling.lessons import NAMES, Lesson
from oddling.persist import load_meta, load_params, save
from oddling.trainer import Trainer

W, H = 1280, 720
VIEW = (820, 720)
PANEL = (460, 720)


def draw_chart(surf, chart: list[float], label: str) -> None:
    pygame.draw.rect(surf, (245, 240, 228), surf.get_rect())
    font = pygame.font.SysFont("consolas", 16)
    surf.blit(font.render(label, True, (23, 33, 28)), (12, 8))
    if len(chart) < 2:
        return
    xs = np.linspace(16, surf.get_width() - 12, len(chart))
    lo, hi = min(chart), max(chart)
    span = max(1e-6, hi - lo)
    pts = [(int(x), int(surf.get_height() - 16 - (y - lo) / span * (surf.get_height() - 40))) for x, y in zip(xs, chart)]
    if len(pts) >= 2:
        pygame.draw.lines(surf, (211, 162, 74), False, pts, 2)


def draw_brain(surf, brain: dict) -> None:
    font = pygame.font.SysFont("consolas", 14)
    surf.fill((245, 240, 228))
    surf.blit(font.render("brain", True, (23, 33, 28)), (12, 8))
    h1 = np.asarray(brain.get("h1", []), dtype=float)
    if h1.size == 0:
        return
    for i, v in enumerate(h1[:32]):
        x, y = 12 + (i % 16) * 22, 32 + (i // 16) * 22
        c = int(np.clip(128 + v * 80, 0, 255))
        pygame.draw.rect(surf, (c, 90, 70), (x, y, 18, 18))


def main(argv: list[str] | None = None) -> int:
    pygame.init()
    pygame.display.set_caption("Oddling")
    screen = pygame.display.set_mode((W, H))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 16)
    n_envs = 64
    if "--cpu" in (argv or sys.argv):
        n_envs = 8
    trainer = Trainer(n_envs=n_envs, n_steps=8)
    meta = load_meta()
    if meta:
        trainer.lesson = Lesson.from_dict(meta.get("lesson", {}))
        trainer.chart = list(meta.get("chart", []))
        trainer.steps = int(meta.get("steps", 0))
        loaded = load_params(trainer.params)
        if loaded is not None:
            trainer.params = loaded
    mj_model = load_mj_model()
    mj_data = mujoco.MjData(mj_model)
    renderer = mujoco.Renderer(mj_model, height=H, width=VIEW[0])
    paused = False
    turbo = 1
    speeds = [1, 2, 4, 8, 16]
    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_SPACE:
                    paused = not paused
                elif ev.key == pygame.K_RIGHTBRACKET:
                    turbo = min(len(speeds) - 1, turbo + 1)
                elif ev.key == pygame.K_LEFTBRACKET:
                    turbo = max(0, turbo - 1)
                elif ev.key == pygame.K_r:
                    trainer.set_lesson(Lesson.default())
            elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                mx, my = ev.pos
                if mx < VIEW[0]:
                    trainer.drop_food(FOOD_HOME)
                else:
                    px = mx - VIEW[0]
                    for i, name in enumerate(NAMES):
                        y = 240 + i * 36
                        if 16 <= px <= 396 and y <= my <= y + 22:
                            t = (px - 16) / 380
                            trainer.set_lesson(trainer.lesson.scaled(name, t))
                    if 16 <= px <= 176 and 348 <= my <= 376:
                        trainer.set_lesson(Lesson.default())
        if not paused:
            trainer.tick(unrolls=speeds[turbo])
        snap = trainer.snapshot()
        mj_data.qpos[:] = snap["qpos"]
        mj_data.qvel[:] = snap["qvel"]
        if mj_model.nmocap > 0:
            mj_data.mocap_pos[0] = snap["food"]
        mujoco.mj_forward(mj_model, mj_data)
        renderer.update_scene(mj_data, camera=0)
        pixels = renderer.render()
        # paint food
        view = pygame.surfarray.make_surface(np.transpose(pixels, (1, 0, 2)))
        screen.blit(view, (0, 0))
        panel = pygame.Surface(PANEL)
        panel.fill((246, 240, 228))
        hud = f"energy {snap['energy']:.1f}  eats {snap['eats']}  steps {trainer.steps}  {speeds[turbo]}x"
        if not snap["alive"]:
            hud += "  collapsed"
        panel.blit(font.render(hud, True, (23, 33, 28)), (16, 16))
        mean = trainer.chart[-1] if trainer.chart else 0.0
        panel.blit(font.render(f"progress {mean:.3f}", True, (23, 33, 28)), (16, 40))
        chart_s = pygame.Surface((430, 140))
        draw_chart(chart_s, trainer.chart, "progress over time")
        panel.blit(chart_s, (16, 64))
        panel.blit(font.render("lessons  (click)", True, (23, 33, 28)), (16, 214))
        for i, name in enumerate(NAMES):
            val = trainer.lesson.as_dict()[name]
            y = 240 + i * 36
            pygame.draw.rect(panel, (200, 190, 170), (16, y, 380, 22))
            pygame.draw.rect(panel, (47, 90, 69), (16, y, int(380 * val), 22))
            panel.blit(font.render(f"{name} {val:.2f}", True, (23, 33, 28)), (20, y + 2))
        pygame.draw.rect(panel, (211, 162, 74), (16, 348, 160, 28))
        panel.blit(font.render("Reset lesson", True, (23, 33, 28)), (24, 352))
        brain_s = pygame.Surface((430, 90))
        draw_brain(brain_s, trainer.brain())
        panel.blit(brain_s, (16, 390))
        help_l = font.render("[ ] turbo  space pause  click field = food", True, (80, 80, 70))
        panel.blit(help_l, (16, 500))
        screen.blit(panel, (VIEW[0], 0))
        pygame.display.flip()
        clock.tick(30)
    save(trainer.lesson, trainer.chart, trainer.steps, trainer.params)
    renderer.close()
    pygame.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
