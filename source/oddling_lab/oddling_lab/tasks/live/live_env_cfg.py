from __future__ import annotations

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_ov.physics import OvPhysxCfg
from isaaclab_physx.physics import PhysxCfg

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.physics import PhysxAutoCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass
from isaaclab.visualizers import VisualizerCfg

from isaaclab_tasks.utils import PresetCfg

from oddling.live import FOOD_HOME
from oddling_lab.assets.critter import CRITTER_CFG, FOOD_CFG


@configclass
class LivePhysicsCfg(PresetCfg):
    isaacsim_physx: PhysxCfg = PhysxCfg(bounce_threshold_velocity=0.2)
    ovphysx: OvPhysxCfg = OvPhysxCfg()
    physx: PhysxAutoCfg = PhysxAutoCfg(isaacsim_physx=isaacsim_physx, ovphysx=ovphysx)
    newton_mjwarp: NewtonCfg = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            njmax=128,
            nconmax=64,
            cone="pyramidal",
            integrator="implicitfast",
            impratio=1,
        ),
        num_substeps=1,
        debug_mode=False,
    )
    default = newton_mjwarp


@configclass
class LiveEnvCfg(DirectRLEnvCfg):
    episode_length_s = 90.0
    decimation = 2
    action_scale = 1.0
    action_space = 8
    observation_space = 23
    state_space = 0

    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation, physics=LivePhysicsCfg())

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1024, env_spacing=6.0, replicate_physics=True, clone_in_fabric=True
    )

    robot = CRITTER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    food = FOOD_CFG.replace(prim_path="{ENV_REGEX_NS}/Food")
    food_home: tuple[float, float, float] = FOOD_HOME
    food_jitter_x: float = 0.35
    food_jitter_y: float = 0.35

    rew_eat: float = 25.0
    rew_alive: float = 0.0
    rew_approach: float = 2.5
    rew_upright: float = 0.005
    rew_dead: float = -2.0

    def __post_init__(self):
        self.sim.default_visualizer_cfg = VisualizerCfg(eye=(4.0, -2.5, 1.8), lookat=(0.4, 0.0, 0.2))
