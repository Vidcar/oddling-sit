import gymnasium as gym

from . import agents

gym.register(
    id="Oddling-Live-Direct",
    entry_point=f"{__name__}.live_env:LiveEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.live_env_cfg:LiveEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LivePPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)
