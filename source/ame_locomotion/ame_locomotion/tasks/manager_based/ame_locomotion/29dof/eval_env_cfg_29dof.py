"""Evaluation-only environment config (used by scripts/rsl_rl/evaluate.py; not used for training or play).

Differences from G1RoughEnvCfg_PLAY:
- num_rows x num_cols tiles of ONE terrain type at a FIXED difficulty, generator seeded;
- env i spawns at the center of tile ((i // num_cols) % num_rows, i % num_cols) (EvalTerrainImporter);
- no camera, no terrain curriculum, 20 s episodes;
- policy height scan uses ElevationMapEval with configurable noise; stage-2 noise by default;
- the illegal-contact termination also records which body touched (same termination result).
"""

import copy

import torch

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from ame_locomotion.tasks.manager_based.ame_locomotion.mdp.eval_observations import ElevationMapEval
from ame_locomotion.tasks.manager_based.ame_locomotion.mdp.eval_terminations import illegal_contact_recorded

from .velocity_env_cfg_29dof import PLAY_SUB_TERRAINS, G1RoughEnvCfg_PLAY

EVAL_TERRAINS = ["steppingstones", "stairs_down", "stairs_up", "stakes"]

# Stage-2 (FINETUNE=True) noise values, copied from ObservationsCfg.PolicyCfg / observations.elevation_map.
STAGE2_HEIGHT_NOISE = {"height_sigma": 0.03, "height_offset": 0.05, "xy_shift": 0.0, "delay_steps": 1}
STAGE2_PROPRIO_NOISE = {"base_ang_vel": 0.2, "projected_gravity": 0.05, "joint_pos": 0.01, "joint_vel": 2.0}


class EvalTerrainImporter(TerrainImporter):
    """Deterministic spawning: env i -> tile ((i // num_cols) % num_rows, i % num_cols), instead of a random row."""

    def _compute_env_origins_curriculum(self, num_envs: int, origins: torch.Tensor) -> torch.Tensor:
        num_rows, num_cols = origins.shape[:2]
        self.max_terrain_level = num_rows
        env_ids = torch.arange(num_envs, device=self.device)
        self.terrain_levels = (env_ids // num_cols) % num_rows
        self.terrain_types = env_ids % num_cols
        return origins[self.terrain_levels, self.terrain_types].clone()


def set_proprio_noise(cfg, enabled: bool):
    """Stage-2 additive uniform noise on proprioceptive policy observations, or none."""
    policy = cfg.observations.policy
    for name, n in STAGE2_PROPRIO_NOISE.items():
        getattr(policy, name).noise = Unoise(n_min=-n, n_max=n) if enabled else None
    policy.enable_corruption = enabled


@configclass
class G1EvalEnvCfg(G1RoughEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 20.0
        # headless evaluation: no camera, no command arrows
        self.scene.visualize_cam = None
        self.commands.base_velocity.debug_vis = False
        # fixed tiles, deterministic spawning
        self.curriculum.terrain_levels = None
        self.scene.terrain.class_type = EvalTerrainImporter
        # policy height scan with configurable noise (critic group unchanged)
        self.observations.policy.height_scan = ObsTerm(
            func=ElevationMapEval,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), **STAGE2_HEIGHT_NOISE},
        )
        set_proprio_noise(self, True)
        # same contact termination, plus a record of the touching body (same params)
        self.terminations.base_contact.func = illegal_contact_recorded


def configure_eval(
    cfg: G1EvalEnvCfg,
    terrain: str,
    num_rows: int,
    num_cols: int,
    difficulty: float,
    seed: int,
    episode_s: float,
    height_noise: dict,
    proprio_noise: bool,
):
    """Apply CLI settings to an eval config."""
    gen = cfg.scene.terrain.terrain_generator
    if terrain != "stakes":  # stakes is the G1RoughEnvCfg_PLAY default
        gen.sub_terrains = copy.deepcopy(PLAY_SUB_TERRAINS[terrain])
    gen.num_rows = num_rows
    gen.num_cols = num_cols
    gen.difficulty_range = (difficulty, difficulty)
    gen.curriculum = False
    gen.seed = seed
    cfg.seed = seed
    cfg.episode_length_s = episode_s
    cfg.observations.policy.height_scan.params.update(height_noise)
    set_proprio_noise(cfg, proprio_noise)
