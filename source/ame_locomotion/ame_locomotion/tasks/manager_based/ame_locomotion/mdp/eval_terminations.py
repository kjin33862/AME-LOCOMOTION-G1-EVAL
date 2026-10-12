"""Evaluation-only termination terms. Not used by any training or play config.

``illegal_contact_recorded`` returns exactly what ``isaaclab.envs.mdp.illegal_contact`` returns and also stores,
per env, which body had the largest contact force when it fired. It has to be recorded here: the env resets
terminated envs inside ``env.step()``, and the contact sensor reset zeroes the force history.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

CONTACT_BODY_ATTR = "eval_contact_body"  # env attribute: contact-sensor body index per env, -1 = none yet


def illegal_contact_recorded(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Same as ``mdp.illegal_contact``; also records the body with the largest force for envs that terminate."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    # same expression as mdp.illegal_contact, split in two
    force = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0]
    terminated = torch.any(force > threshold, dim=1)

    if not hasattr(env, CONTACT_BODY_ATTR):
        setattr(env, CONTACT_BODY_ATTR, torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device))
    body_ids = torch.as_tensor(sensor_cfg.body_ids, dtype=torch.long, device=env.device)
    strongest = body_ids[torch.argmax(force, dim=1)]
    record = getattr(env, CONTACT_BODY_ATTR)
    record[terminated] = strongest[terminated]
    return terminated
