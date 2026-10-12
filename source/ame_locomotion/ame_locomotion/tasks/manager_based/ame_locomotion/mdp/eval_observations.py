"""Evaluation-only observation terms. Not used by any training or play config.

``ElevationMapEval`` reproduces ``observations.elevation_map`` (same geometry and clipping), with two differences:

- the per-env height offset and x-y shift are sampled in :meth:`reset`, which the observation manager also calls on
  the initial reset, so they apply during an env's FIRST episode (``elevation_map`` keeps them at zero until the
  first ``reset_buf``);
- noise magnitudes and the map update delay are parameters instead of constants.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import RayCaster
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import ObservationTermCfg


class ElevationMapEval(ManagerTermBase):
    """Elevation map with configurable noise.

    Params:
        sensor_cfg: The height scanner.
        height_sigma: Std of per-cell Gaussian height noise [m] (stage-2 training: 0.03).
        height_offset: Per-env, per-episode height offset ~ U(-height_offset, height_offset) [m] (stage-2: 0.05).
        xy_shift: Per-env, per-episode x-y shift ~ U(-xy_shift, xy_shift) [m] added to the x/y coordinate
            channels (commented out in training code: 0.03).
        delay_steps: Refresh the map every ``delay_steps`` policy steps and on reset; 0 or 1 disables
            (commented out in training code: 5, i.e. 10 Hz).
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._offset = torch.zeros((self.num_envs, 1), device=self.device)
        self._shift = torch.zeros((self.num_envs, 2), device=self.device)
        self._last_map: torch.Tensor | None = None
        self.reset()

    def reset(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        n = len(env_ids)
        height_offset = self.cfg.params["height_offset"]
        xy_shift = self.cfg.params["xy_shift"]
        self._offset[env_ids] = (torch.rand((n, 1), device=self.device) * 2.0 - 1.0) * height_offset
        self._shift[env_ids] = (torch.rand((n, 2), device=self.device) * 2.0 - 1.0) * xy_shift

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg,
        height_sigma: float,
        height_offset: float,
        xy_shift: float,
        delay_steps: int,
    ) -> torch.Tensor:
        # same transform as observations.elevation_map
        sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
        relative_pos_w = sensor.data.ray_hits_w - sensor.data.pos_w.unsqueeze(1)
        sensor_quat = sensor.data.quat_w
        N, B, _ = relative_pos_w.shape
        if getattr(sensor.cfg, "ray_alignment", "base") == "yaw":
            sensor_quat = yaw_quat(sensor_quat)
        sensor_quat_expanded = sensor_quat.unsqueeze(1).expand(N, B, 4).reshape(N * B, 4).to(torch.float)
        sensor_coords = quat_apply_inverse(sensor_quat_expanded, relative_pos_w.reshape(N * B, 3)).reshape(N, B, 3)
        sensor_coords = torch.nan_to_num(sensor_coords)

        # noise, in the order of the training code: x-y shift, Gaussian height noise, height offset
        if xy_shift > 0.0:
            sensor_coords[..., :2] += self._shift.unsqueeze(1)
        if height_sigma > 0.0:
            sensor_coords[..., 2] += torch.randn_like(sensor_coords[..., 2]) * height_sigma
        if height_offset > 0.0:
            sensor_coords[..., 2] += self._offset

        # clip height values (same as training)
        sensor_coords[..., 2] = torch.clamp(sensor_coords[..., 2], min=-1.2, max=0.0)
        current_map = sensor_coords.reshape(N, B * 3)

        if delay_steps <= 1:
            return current_map
        # update delay: refresh every `delay_steps` steps; reset envs have episode_length_buf == 0
        if self._last_map is None or self._last_map.shape != current_map.shape:
            self._last_map = current_map.clone()
        update_mask = env.episode_length_buf % delay_steps == 0
        self._last_map[update_mask] = current_map[update_mask]
        return self._last_map.clone()
