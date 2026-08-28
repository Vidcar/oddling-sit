from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.sim as sim_utils
from isaaclab import cloner
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_apply, quat_from_euler_xyz

from oddling.live import DRAIN, EAT_ENERGY, EAT_LOCK, EAT_RADIUS, FOOD_CLEAR, START_ENERGY

if TYPE_CHECKING:
    from oddling_lab.tasks.live.live_env_cfg import LiveEnvCfg


class LiveEnv(DirectRLEnv):
    """Flat-field eat-or-die. No walk-forward score."""

    cfg: LiveEnvCfg

    def __init__(self, cfg: LiveEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.action_scale = self.cfg.action_scale
        n = self.num_envs
        device = self.device
        self.energy = torch.full((n,), START_ENERGY, device=device)
        self.eats = torch.zeros(n, dtype=torch.int32, device=device)
        self.food_pos = torch.zeros((n, 3), device=device)
        self.food_pos[:] = torch.tensor(self.cfg.food_home, device=device)
        self.prev_dist = torch.full((n,), -1.0, device=device)
        self.eat_lock = torch.zeros(n, dtype=torch.int32, device=device)
        self.starved = torch.zeros(n, dtype=torch.bool, device=device)
        self.ate = torch.zeros(n, dtype=torch.bool, device=device)
        self.approach = torch.zeros(n, device=device)
        self._mouth_ids, _ = self.robot.find_bodies("mouth.*")
        if len(self._mouth_ids) == 0:
            self._mouth_ids, _ = self.robot.find_bodies("torso")

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        self.food = RigidObject(self.cfg.food)
        spawn_ground_plane(
            prim_path="/World/ground",
            cfg=GroundPlaneCfg(color=(0.18, 0.35, 0.27)),
        )
        src, dest = "/World/envs/env_0", "/World/envs/env_{}"
        pos = cloner.grid_transforms(self.scene.num_envs, self.scene.cfg.env_spacing, device=self.device)[0]
        global_paths = ("/World/ground",)
        plan = cloner.clone_plan_from_env_0(src, dest, self.scene.num_envs, self.device, pos, global_paths=global_paths)
        cloner.replicate(plan, stage=self.scene.stage)
        if "physx" in self.scene.physics_backend:
            self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["food"] = self.food
        light_cfg = sim_utils.DistantLightCfg(intensity=2000.0, color=(1.0, 0.95, 0.85))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = self.action_scale * actions.clone()
        self.actions = self.actions.clamp(-1.0, 1.0)

    def _apply_action(self) -> None:
        self.robot.set_joint_effort_target_index(target=self.actions)

    def _mouth_world(self) -> torch.Tensor:
        return self.robot.data.body_pos_w.torch[:, self._mouth_ids[0]]

    def _write_food(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        pose = torch.zeros((env_ids.shape[0], 7), device=self.device)
        pose[:, :3] = self.food_pos[env_ids] + self.scene.env_origins[env_ids]
        pose[:, 3] = 1.0
        self.food.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)

    def _jitter_food(self, env_ids: torch.Tensor) -> None:
        n = env_ids.shape[0]
        mouth = self._mouth_world()[env_ids] - self.scene.env_origins[env_ids]
        dist = FOOD_CLEAR + torch.rand(n, device=self.device) * self.cfg.food_jitter_x
        ang = (torch.rand(n, device=self.device) - 0.5) * 1.2
        self.food_pos[env_ids, 0] = mouth[:, 0] + dist * torch.cos(ang)
        self.food_pos[env_ids, 1] = mouth[:, 1] + dist * torch.sin(ang)
        self.food_pos[env_ids, 2] = 0.12

    def _get_observations(self) -> dict:
        jp = self.robot.data.joint_pos.torch
        jv = self.robot.data.joint_vel.torch
        root_quat = self.robot.data.root_quat_w.torch
        down = torch.zeros((self.num_envs, 3), device=self.device)
        down[:, 2] = -1.0
        # up is -gravity in body? use world +Z through quat
        z_axis = torch.zeros_like(down)
        z_axis[:, 2] = 1.0
        up = quat_apply(root_quat, z_axis)
        mouth = self._mouth_world()
        rel = (self.food_pos + self.scene.env_origins) - mouth
        e = (self.energy / START_ENERGY).unsqueeze(-1)
        obs = torch.cat((jp, jv, up, rel, e), dim=-1)
        return {"policy": obs}

    def _tick_live(self) -> None:
        mouth = self._mouth_world()
        food_w = self.food_pos + self.scene.env_origins
        dist = torch.linalg.norm(mouth - food_w, dim=-1)
        ate = (dist < EAT_RADIUS) & (self.eat_lock <= 0)
        self.eat_lock = torch.where(self.eat_lock > 0, self.eat_lock - 1, self.eat_lock)
        self.eat_lock = torch.where(ate, torch.full_like(self.eat_lock, EAT_LOCK), self.eat_lock)
        self.energy = self.energy - DRAIN + ate.float() * EAT_ENERGY
        self.eats = self.eats + ate.to(torch.int32)
        if bool(ate.any()):
            self._jitter_food(ate.nonzero(as_tuple=False).squeeze(-1))
            self._write_food(ate.nonzero(as_tuple=False).squeeze(-1))
            dist = torch.linalg.norm(self._mouth_world() - (self.food_pos + self.scene.env_origins), dim=-1)
        self.starved = self.energy <= 0.0
        self.energy = torch.where(self.starved, torch.zeros_like(self.energy), self.energy)
        have_prev = self.prev_dist >= 0.0
        self.approach = torch.where(have_prev, self.prev_dist - dist, torch.zeros_like(dist))
        self.prev_dist = dist.detach()
        self.ate = ate
        self.dist = dist

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._tick_live()
        time_out = self.episode_length_buf >= self.max_episode_length
        return self.starved, time_out

    def _get_rewards(self) -> torch.Tensor:
        z_axis = torch.zeros((self.num_envs, 3), device=self.device)
        z_axis[:, 2] = 1.0
        up = quat_apply(self.robot.data.root_quat_w.torch, z_axis)
        upright = up[:, 2].clamp(min=0.0, max=1.0)
        rew = (
            self.cfg.rew_eat * self.ate.float()
            + self.cfg.rew_alive * (~self.starved).float()
            + self.cfg.rew_approach * self.approach
            + self.cfg.rew_upright * upright
            + self.cfg.rew_dead * self.starved.float()
        )
        self.extras.setdefault("log", {})
        self.extras["log"]["eats_mean"] = self.eats.float().mean().item()
        self.extras["log"]["energy_mean"] = self.energy.mean().item()
        return rew

    def _reset_life(self, env_ids: torch.Tensor) -> None:
        self.energy[env_ids] = START_ENERGY
        self.eats[env_ids] = 0
        self.prev_dist[env_ids] = -1.0
        self.eat_lock[env_ids] = 0
        home = torch.tensor(self.cfg.food_home, device=self.device)
        self.food_pos[env_ids] = home
        self._write_food(env_ids)
        self._reset_body(env_ids)

    def _reset_body(self, env_ids: torch.Tensor) -> None:
        n = env_ids.shape[0]
        joint_pos = self.robot.data.default_joint_pos.torch[env_ids].clone()
        joint_pos += 0.08 * (2.0 * torch.rand_like(joint_pos) - 1.0)
        joint_vel = self.robot.data.default_joint_vel.torch[env_ids].clone()
        root_pose = self.robot.data.default_root_pose.torch[env_ids].clone()
        yaw = 0.4 * (2.0 * torch.rand(n, device=self.device) - 1.0)
        root_pose[:, 3:7] = quat_from_euler_xyz(
            torch.zeros(n, device=self.device),
            torch.zeros(n, device=self.device),
            yaw,
        )
        root_pose[:, :3] += self.scene.env_origins[env_ids]
        root_vel = self.robot.data.default_root_vel.torch[env_ids].clone()
        self.robot.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
        self.robot.write_root_velocity_to_sim_index(root_velocity=root_vel, env_ids=env_ids)
        self.robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids)

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self._reset_life(ids)
