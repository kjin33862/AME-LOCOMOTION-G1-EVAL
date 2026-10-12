import gymnasium as gym
from ame_locomotion.tasks.manager_based.ame_locomotion import agents

gym.register(
    id="AME-G1-29DOF-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_29dof:G1RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ame_rsl_rl_ppo_cfg:G1AMEPPORunnerCfg",
        # "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
    },
)

gym.register(
    id="AME-G1-29DOF-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_29dof:G1RoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ame_rsl_rl_ppo_cfg:G1AMEPPORunnerCfg",
        # "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
    },
)

# TEMP (ame2 play)
gym.register(
    id="AME-G1-29DOF-Global-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_29dof:G1RoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ame_rsl_rl_ppo_cfg:G1AMEGlobalPPORunnerCfg",
    },
)

# Play variants per terrain (stakes = the default *-Play-v0 tasks above)
for _suffix, _env_cfg in [
    ("StairsDown", "G1RoughEnvCfg_PLAY_STAIRS_DOWN"),
    ("StairsUp", "G1RoughEnvCfg_PLAY_STAIRS_UP"),
    ("SteppingStones", "G1RoughEnvCfg_PLAY_STEPPINGSTONES"),
]:
    for _prefix, _agent_cfg in [("AME-G1-29DOF", "G1AMEPPORunnerCfg"), ("AME-G1-29DOF-Global", "G1AMEGlobalPPORunnerCfg")]:
        gym.register(
            id=f"{_prefix}-Play-{_suffix}-v0",
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_29dof:{_env_cfg}",
                "rsl_rl_cfg_entry_point": f"{agents.__name__}.ame_rsl_rl_ppo_cfg:{_agent_cfg}",
            },
        )
