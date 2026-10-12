"""Quantitative evaluation of an AME checkpoint (headless, one episode per env).

Each env spawns at the center of its own terrain tile and is commanded 1.0 m/s forward with heading 0.
Outcome per env, from its FIRST episode only (auto-resets afterwards are ignored):
    success: base xy displacement from spawn >= --success_dist before a fail
    fail:    illegal-contact termination, or a fall (tilt > --tilt_deg, or base height above the supporting
             surface < --fall_height; optionally base z < spawn z - --spawn_drop), held for --fall_hold_s
    stuck:   neither by the end of the episode

Validity guard: the run is aborted and summary.json is written with "valid": false if PhysX reports a terrain
mesh "cooking failure" (robots then fall through the ground) or if more than 50% of envs fail by "height"
within the first second. Valid runs have "valid": true.

Visual check mode (--viz): opens the Isaac Sim window, defaults to a 3x3 grid (9 robots), runs at real-time speed
(--viz_speed 0.25 = 4x slower), prints one line per env the moment it gets its outcome, and shows a sphere above each
robot (white running, green success, red fail, yellow stuck). After the last outcome the scene stays frozen until
the window is closed. The outcome rules are the same as headless; results are saved too (folder tag "viz" unless
--tag is given).

Example:
    python scripts/rsl_rl/evaluate.py --arch ame1 --terrain steppingstones --num_envs 100 --seed 0
    python scripts/rsl_rl/evaluate.py --arch ame1 --terrain steppingstones --viz
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

parser = argparse.ArgumentParser(description="Evaluate an RSL-RL AME checkpoint.")
parser.add_argument("--arch", type=str, default="ame1", choices=["ame1", "ame2"],
                    help="Policy architecture: ame1 (G1AMEPPORunnerCfg) or ame2 (attach_global=True).")
parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path. Default: pretrained/<arch>.pt.")
parser.add_argument("--terrain", type=str, default="steppingstones",
                    choices=["steppingstones", "stairs_down", "stairs_up", "stakes"])
parser.add_argument("--num_envs", type=int, default=None, help="Default: 100 (9 with --viz).")
parser.add_argument("--num_rows", type=int, default=None, help="Default: 10 (3 with --viz).")
parser.add_argument("--num_cols", type=int, default=None, help="Default: 10 (3 with --viz).")
parser.add_argument("--difficulty", type=float, default=0.5, help="Fixed terrain difficulty for all tiles.")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--episode_s", type=float, default=20.0)
# noise
parser.add_argument("--noise", type=str, default="stage2", choices=["stage2", "off"],
                    help="Preset. stage2: height sigma 0.03, offset 0.05, proprio noise on. off: no noise.")
parser.add_argument("--height_sigma", type=float, default=None, help="Override preset [m].")
parser.add_argument("--height_offset", type=float, default=None, help="Override preset [m].")
parser.add_argument("--xy_shift", type=float, default=0.0, help="Per-episode x-y map shift [m] (training code: 0.03).")
parser.add_argument("--map_delay_steps", type=int, default=1, help="Map refresh period in steps; 5 = 10 Hz. 1 = off.")
parser.add_argument("--proprio_noise", type=str, default="preset", choices=["preset", "on", "off"])
# outcome rules
parser.add_argument("--success_dist", type=float, default=4.0)
parser.add_argument("--tilt_deg", type=float, default=60.0)
parser.add_argument("--fall_height", type=float, default=0.4, help="Min base height above supporting surface [m].")
parser.add_argument("--support_radius", type=float, default=0.3, help="XY radius around base for the surface [m].")
parser.add_argument("--fall_hold_s", type=float, default=0.2)
parser.add_argument("--spawn_drop_rule", action="store_true", default=False,
                    help="Also fail if base z drops more than --spawn_drop below spawn (wrong on stairs_down).")
parser.add_argument("--spawn_drop", type=float, default=0.5)
# run control / output
parser.add_argument("--max_steps", type=int, default=None, help="Stop early (smoke / memory tests).")
parser.add_argument("--out_root", type=str, default=os.path.join(REPO_ROOT, "eval_results"))
parser.add_argument("--tag", type=str, default="", help="Suffix for the output folder name.")
parser.add_argument("--viz", action="store_true", default=False,
                    help="Visual check: Isaac Sim window, 3x3 grid, real time, outcome lines and markers.")
parser.add_argument("--viz_speed", type=float, default=1.0,
                    help="--viz only: playback speed vs real time (0.25 = 4x slower). Does not change the results.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.viz_speed <= 0:
    parser.error("--viz_speed must be > 0")
# grid defaults: 10x10 / 100 envs headless, 3x3 / 9 envs with --viz
for name, headless_default, viz_default in (("num_envs", 100, 9), ("num_rows", 10, 3), ("num_cols", 10, 3)):
    if getattr(args_cli, name) is None:
        setattr(args_cli, name, viz_default if args_cli.viz else headless_default)
if args_cli.viz and not args_cli.tag:
    args_cli.tag = "viz"
# headless unless --viz; no cameras
args_cli.headless = not args_cli.viz
args_cli.enable_cameras = False

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import csv
import importlib
import json
import math
import subprocess
import time
from datetime import datetime

import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from ame_locomotion.tasks.manager_based.ame_locomotion.agents.ame_rsl_rl_ppo_cfg import (
    G1AMEGlobalPPORunnerCfg,
    G1AMEPPORunnerCfg,
)

from ame_locomotion.tasks.manager_based.ame_locomotion.mdp.eval_terminations import CONTACT_BODY_ATTR

# "29dof" is not a valid identifier, so import by string
eval_cfg_module = importlib.import_module("ame_locomotion.tasks.manager_based.ame_locomotion.29dof.eval_env_cfg_29dof")

PENDING, SUCCESS, FAIL, STUCK = 0, 1, 2, 3
OUTCOME_NAMES = {PENDING: "pending", SUCCESS: "success", FAIL: "fail", STUCK: "stuck"}
FAIL_REASONS = {0: "", 1: "contact", 2: "tilt", 3: "height", 4: "spawn_drop"}
COOKING_FAILURE = "cooking failure"
EARLY_WINDOW_S = 1.0  # validity guard: window for early height fails
EARLY_HEIGHT_FRAC = 0.5  # validity guard: max fraction of envs failing by height in that window


def resolve_noise(args) -> tuple[dict, bool]:
    stage2 = args.noise == "stage2"
    height_noise = dict(eval_cfg_module.STAGE2_HEIGHT_NOISE) if stage2 else {
        "height_sigma": 0.0, "height_offset": 0.0, "xy_shift": 0.0, "delay_steps": 1
    }
    if args.height_sigma is not None:
        height_noise["height_sigma"] = args.height_sigma
    if args.height_offset is not None:
        height_noise["height_offset"] = args.height_offset
    height_noise["xy_shift"] = args.xy_shift
    height_noise["delay_steps"] = args.map_delay_steps
    proprio = stage2 if args.proprio_noise == "preset" else args.proprio_noise == "on"
    return height_noise, proprio


def noise_tag(height_noise: dict, proprio: bool) -> str:
    if not proprio and height_noise["height_sigma"] == 0 and height_noise["height_offset"] == 0 \
            and height_noise["xy_shift"] == 0 and height_noise["delay_steps"] <= 1:
        return "noiseoff"
    return (f"h{height_noise['height_sigma']:g}-o{height_noise['height_offset']:g}"
            f"-xy{height_noise['xy_shift']:g}-d{height_noise['delay_steps']}-p{int(proprio)}")


def git_info() -> dict:
    try:
        commit = subprocess.check_output(["git", "-C", REPO_ROOT, "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", REPO_ROOT, "status", "--porcelain", "-uno"], text=True).strip()
        return {"commit": commit, "dirty_files": dirty.splitlines()}
    except Exception as e:  # git missing or not a repo
        return {"error": str(e)}


def kit_log_path() -> str | None:
    """Path of this process's Kit log file (PhysX cooking warnings go there)."""
    try:
        import carb

        path = carb.settings.get_settings().get("/log/file")
        return path if path and os.path.isfile(path) else None
    except Exception:
        return None


def cooking_failures(path: str | None) -> list[str] | None:
    """Lines of the Kit log that report a mesh cooking failure; None if the log is unavailable."""
    if path is None:
        return None
    with open(path, errors="replace") as f:
        return [line.strip() for line in f if COOKING_FAILURE in line]


def make_outcome_markers() -> "VisualizationMarkers":
    """One sphere per env; prototype index = outcome code (PENDING white, SUCCESS green, FAIL red, STUCK yellow)."""
    import isaaclab.sim as sim_utils
    from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

    def sphere(rgb):
        return sim_utils.SphereCfg(radius=0.12, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=rgb))

    return VisualizationMarkers(VisualizationMarkersCfg(prim_path="/Visuals/EvalOutcome", markers={
        "running": sphere((1.0, 1.0, 1.0)),
        "success": sphere((0.1, 0.8, 0.1)),
        "fail": sphere((0.9, 0.1, 0.1)),
        "stuck": sphere((1.0, 0.85, 0.0)),
    }))


def rate(k: int, n: int) -> dict:
    p = k / n if n > 0 else float("nan")
    se = math.sqrt(p * (1.0 - p) / n) if n > 0 else float("nan")
    return {"count": k, "rate": p, "se": se}


def main():
    height_noise, proprio = resolve_noise(args_cli)
    checkpoint = args_cli.checkpoint or os.path.join(REPO_ROOT, "pretrained", f"{args_cli.arch}.pt")
    checkpoint = os.path.abspath(os.path.expanduser(checkpoint))

    # -- configs
    env_cfg = eval_cfg_module.G1EvalEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    eval_cfg_module.configure_eval(
        env_cfg, args_cli.terrain, args_cli.num_rows, args_cli.num_cols, args_cli.difficulty,
        args_cli.seed, args_cli.episode_s, height_noise, proprio,
    )
    agent_cfg = G1AMEGlobalPPORunnerCfg() if args_cli.arch == "ame2" else G1AMEPPORunnerCfg()
    agent_cfg.seed = args_cli.seed
    agent_cfg.device = env_cfg.sim.device

    # -- env and policy (same loading as play.py)
    env = RslRlVecEnvWrapper(ManagerBasedRLEnv(cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    print(f"[INFO] Loading model checkpoint from: {checkpoint}")
    runner.load(checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    ckpt_iter = torch.load(checkpoint, map_location="cpu", weights_only=False).get("iter")

    base_env = env.unwrapped
    robot = base_env.scene["robot"]
    scanner = base_env.scene["height_scanner"]
    terrain = base_env.scene.terrain
    contact_body_names = base_env.scene["contact_forces"].body_names
    num_envs, device, dt = base_env.num_envs, base_env.device, base_env.step_dt

    # -- per-env buffers
    spawn_pos = robot.data.root_pos_w.clone()
    active = torch.ones(num_envs, dtype=torch.bool, device=device)
    outcome = torch.full((num_envs,), PENDING, dtype=torch.long, device=device)
    fail_reason = torch.zeros(num_envs, dtype=torch.long, device=device)
    fail_time = torch.full((num_envs,), float("nan"), device=device)
    contact_body = torch.full((num_envs,), -1, dtype=torch.long, device=device)
    time_to_success = torch.full((num_envs,), float("nan"), device=device)
    max_forward = torch.zeros(num_envs, device=device)
    max_xy_disp = torch.zeros(num_envs, device=device)
    err_sum = torch.zeros(num_envs, device=device)
    height_sum = torch.zeros(num_envs, device=device)
    n_tracked = torch.zeros(num_envs, dtype=torch.long, device=device)
    tilt_cnt = torch.zeros(num_envs, dtype=torch.long, device=device)
    low_cnt = torch.zeros(num_envs, dtype=torch.long, device=device)
    drop_cnt = torch.zeros(num_envs, dtype=torch.long, device=device)
    hold_steps = max(1, round(args_cli.fall_hold_s / dt))
    cos_tilt = math.cos(math.radians(args_cli.tilt_deg))
    episode_steps = base_env.max_episode_length
    step_limit = episode_steps + 5 if args_cli.max_steps is None else args_cli.max_steps

    # -- visual check mode: env -> tile map, camera over the grid, outcome markers
    tile_rows_t, tile_cols_t = terrain.terrain_levels.tolist(), terrain.terrain_types.tolist()
    if args_cli.viz:
        print("[VIZ] env -> tile (row, col), spawn x, y [m].  Robots walk toward +x (the red axis).")
        for i in range(num_envs):
            print(f"[VIZ]   env {i:3d} -> tile ({tile_rows_t[i]},{tile_cols_t[i]})  spawn "
                  f"({float(spawn_pos[i, 0]):6.1f}, {float(spawn_pos[i, 1]):6.1f})")
        center = spawn_pos.mean(dim=0).tolist()
        size = 8.0 * max(args_cli.num_rows, args_cli.num_cols)
        base_env.sim.set_camera_view(eye=(center[0] - 0.6 * size, center[1] - 0.9 * size, center[2] + 0.7 * size),
                                     target=tuple(center))
        markers = make_outcome_markers()
        marker_pos = spawn_pos.clone()
        last_disp_x = torch.zeros(num_envs, device=device)  # contact-ended envs are already reset: use last step

    # -- output folder (created now so that an aborted run still leaves an INVALID summary)
    tag = noise_tag(height_noise, proprio)
    run_name = f"{datetime.now():%Y%m%d-%H%M%S}_{os.path.splitext(os.path.basename(checkpoint))[0]}_{args_cli.terrain}_{tag}"
    if args_cli.tag:
        run_name += f"_{args_cli.tag}"
    out_dir = os.path.join(os.path.expanduser(args_cli.out_root), run_name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "args.json"), "w") as f:
        json.dump({"argv": sys.argv, **vars(args_cli)}, f, indent=2, default=str)

    # -- validity guard
    kit_log = kit_log_path()
    early_steps = round(EARLY_WINDOW_S / dt)

    def validity(step: int) -> dict:
        cooking = cooking_failures(kit_log)
        early_height = int(((fail_reason == 3) & (fail_time <= EARLY_WINDOW_S + 1e-6)).sum())
        reason = None
        if cooking:
            reason = f"terrain {COOKING_FAILURE} reported by PhysX ({len(cooking)} line(s) in the Kit log)"
        elif step >= early_steps and early_height > EARLY_HEIGHT_FRAC * num_envs:
            reason = (f"{early_height}/{num_envs} envs failed by 'height' within the first {EARLY_WINDOW_S:g} s "
                      f"(> {EARLY_HEIGHT_FRAC:.0%}); the robots probably fell through the terrain")
        return {
            "valid": reason is None,
            "invalid_reason": reason,
            "kit_log": kit_log,
            "cooking_check": "unavailable" if cooking is None else ("failed" if cooking else "ok"),
            "cooking_lines": (cooking or [])[:5],
            "early_height_fails": early_height,
            "checked_at_step": step,
        }

    def abort_if_invalid(step: int) -> dict:
        check = validity(step)
        if check["valid"]:
            return check
        summary = {
            "valid": False,
            "invalid_reason": check["invalid_reason"],
            "validity_check": check,
            "checkpoint": checkpoint,
            "arch": args_cli.arch,
            "terrain": args_cli.terrain,
            "seed": args_cli.seed,
            "num_envs": num_envs,
            "tiles_total": args_cli.num_rows * args_cli.num_cols,
            "height_noise": height_noise,
            "proprio_noise": proprio,
            "tag": args_cli.tag,
            "steps_simulated": step,
        }
        with open(os.path.join(out_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[ERROR] INVALID RUN: {check['invalid_reason']}", flush=True)
        print(f"[ERROR] Marked invalid in: {os.path.join(out_dir, 'summary.json')}", flush=True)
        env.close()
        simulation_app.close()
        sys.exit(2)

    obs = env.get_observations()
    step = 0
    t_start = time.time()
    with torch.inference_mode():
        while simulation_app.is_running() and active.any() and step < step_limit:
            t_step = time.time()
            prev_outcome = outcome.clone()
            actions, *_ = policy(obs)
            obs, _, dones, extras = env.step(actions)
            step += 1

            # envs whose episode ended this step: their state is already reset, so use the outcome only
            done = dones.bool()
            time_out = extras["time_outs"].bool()
            ended = active & done
            outcome[ended & ~time_out] = FAIL
            fail_reason[ended & ~time_out] = 1
            fail_time[ended & ~time_out] = step * dt
            contact_body[ended & ~time_out] = getattr(base_env, CONTACT_BODY_ATTR)[ended & ~time_out]
            outcome[ended & time_out] = STUCK
            active &= ~done

            # still-running envs
            still_running_after_contact = active.clone()
            pos = robot.data.root_pos_w
            disp = pos[:, :2] - spawn_pos[:, :2]
            xy_dist = torch.linalg.norm(disp, dim=-1)
            max_forward = torch.where(active, torch.maximum(max_forward, disp[:, 0]), max_forward)
            max_xy_disp = torch.where(active, torch.maximum(max_xy_disp, xy_dist), max_xy_disp)

            vel_yaw = quat_apply_inverse(yaw_quat(robot.data.root_quat_w), robot.data.root_lin_vel_w)
            cmd = base_env.command_manager.get_command("base_velocity")
            err_sum += torch.where(active, (vel_yaw[:, 0] - cmd[:, 0]).abs(), 0.0)

            # base height above the supporting surface: highest ray hit within support_radius of the base
            hits = scanner.data.ray_hits_w
            near = torch.linalg.norm(hits[..., :2] - pos[:, None, :2], dim=-1) <= args_cli.support_radius
            hit_z = torch.where(torch.isfinite(hits[..., 2]), hits[..., 2], -torch.inf)
            support_z = torch.where(near, hit_z, -torch.inf).max(dim=1).values
            support_z = torch.where(torch.isfinite(support_z), support_z, hit_z.max(dim=1).values)
            height = pos[:, 2] - support_z
            height_sum += torch.where(active, height, 0.0)
            n_tracked += active.long()

            # success first, then falls (falls must persist for hold_steps)
            succeeded = active & (xy_dist >= args_cli.success_dist)
            outcome[succeeded] = SUCCESS
            time_to_success[succeeded] = step * dt
            active &= ~succeeded

            tilted = -robot.data.projected_gravity_b[:, 2] < cos_tilt
            tilt_cnt = torch.where(active & tilted, tilt_cnt + 1, 0)
            low_cnt = torch.where(active & (height < args_cli.fall_height), low_cnt + 1, 0)
            if args_cli.spawn_drop_rule:
                dropped = pos[:, 2] < spawn_pos[:, 2] - args_cli.spawn_drop
                drop_cnt = torch.where(active & dropped, drop_cnt + 1, 0)
            for reason, cnt in ((2, tilt_cnt), (3, low_cnt), (4, drop_cnt)):
                fell = active & (cnt >= hold_steps)
                outcome[fell] = FAIL
                fail_reason[fell] = reason
                fail_time[fell] = step * dt
                active &= ~fell

            if step in (1, early_steps):
                abort_if_invalid(step)

            if args_cli.viz:
                contact_now = ended & ~time_out
                x_now = torch.where(contact_now, last_disp_x, disp[:, 0])
                for i in (outcome != prev_outcome).nonzero().flatten().tolist():
                    line = (f"[t={step * dt:5.2f} s] env {i} (tile {tile_rows_t[i]},{tile_cols_t[i]})  "
                            f"{OUTCOME_NAMES[int(outcome[i])].upper():7s}")
                    if outcome[i] == FAIL:
                        line += f"  reason={FAIL_REASONS[int(fail_reason[i])]}"
                        if contact_body[i] >= 0:
                            line += f"  body={contact_body_names[int(contact_body[i])]}"
                    print(line + f"  x={float(x_now[i]):.1f} m", flush=True)
                last_disp_x = torch.where(active, disp[:, 0], last_disp_x)
                marker_pos = torch.where(still_running_after_contact[:, None], pos, marker_pos)
                markers.visualize(translations=marker_pos + torch.tensor([0.0, 0.0, 0.8], device=device),
                                  marker_indices=outcome)
                sleep_s = dt / args_cli.viz_speed - (time.time() - t_step)
                if sleep_s > 0:
                    time.sleep(sleep_s)

            if step % 100 == 0:
                print(f"[INFO] step {step}/{episode_steps}  active={int(active.sum())}  "
                      f"success={int((outcome == SUCCESS).sum())}  fail={int((outcome == FAIL).sum())}")

    truncated = bool(active.any())  # only if --max_steps cut the run short
    validity_check = abort_if_invalid(max(step, early_steps))
    outcome[active] = STUCK
    wall_time = time.time() - t_start

    # -- per-env table
    tile_rows = terrain.terrain_levels.cpu().tolist()
    tile_cols = terrain.terrain_types.cpu().tolist()
    try:
        friction = robot.root_physx_view.get_material_properties()[..., 0].mean(dim=-1).tolist()
    except Exception:
        friction = [float("nan")] * num_envs
    mean_err = (err_sum / n_tracked.clamp(min=1)).cpu()
    mean_height = (height_sum / n_tracked.clamp(min=1)).cpu()
    outcome_c, reason_c = outcome.cpu().tolist(), fail_reason.cpu().tolist()
    tracked_c = n_tracked.cpu().tolist()
    body_c = contact_body.cpu().tolist()
    rows = []
    for i in range(num_envs):
        survivor = outcome_c[i] != FAIL and tracked_c[i] > 0
        rows.append({
            "env_id": i,
            "tile_row": tile_rows[i],
            "tile_col": tile_cols[i],
            "outcome": OUTCOME_NAMES[outcome_c[i]],
            "fail_reason": FAIL_REASONS[reason_c[i]],
            "contact_body": contact_body_names[body_c[i]] if body_c[i] >= 0 else "",
            "fail_time_s": float(fail_time[i]),
            "time_to_success_s": float(time_to_success[i]),
            "max_forward_m": float(max_forward[i]),
            "max_xy_disp_m": float(max_xy_disp[i]),
            "mean_abs_vx_err": float(mean_err[i]) if survivor else float("nan"),
            "tracked_steps": tracked_c[i],
            "mean_height_above_support_m": float(mean_height[i]),
            "static_friction": float(friction[i]),
        })

    # -- summary
    n = num_envs
    counts = {name: sum(r["outcome"] == name for r in rows) for name in ("success", "fail", "stuck")}
    succ_times = [r["time_to_success_s"] for r in rows if r["outcome"] == "success"]
    surv_errs = [r["mean_abs_vx_err"] for r in rows if r["outcome"] != "fail" and not math.isnan(r["mean_abs_vx_err"])]
    heights = sorted(r["mean_height_above_support_m"] for r in rows if r["tracked_steps"] > 0)

    def mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    summary = {
        "valid": True,
        "validity_check": validity_check,
        "checkpoint": checkpoint,
        "checkpoint_iter": ckpt_iter,
        "arch": args_cli.arch,
        "terrain": args_cli.terrain,
        "difficulty": args_cli.difficulty,
        "seed": args_cli.seed,
        "num_envs": n,
        "tiles_total": args_cli.num_rows * args_cli.num_cols,
        "tiles_distinct_used": len(set(zip(tile_rows, tile_cols))),
        "height_noise": height_noise,
        "proprio_noise": proprio,
        "tag": args_cli.tag,
        "success": rate(counts["success"], n),
        "fail": rate(counts["fail"], n),
        "stuck": rate(counts["stuck"], n),
        "fail_reasons": {r: sum(row["fail_reason"] == r for row in rows) for r in ("contact", "tilt", "height", "spawn_drop")},
        "mean_time_to_success_s": mean(succ_times),
        "mean_max_forward_m": mean([r["max_forward_m"] for r in rows]),
        "mean_abs_vx_err_survivors": mean(surv_errs),
        "n_survivors": len(surv_errs),
        "median_height_above_support_m": heights[len(heights) // 2] if heights else float("nan"),
        "steps_simulated": step,
        "episode_steps": episode_steps,
        "step_dt": dt,
        "truncated_by_max_steps": truncated,
        "wall_time_s": wall_time,
        "torch_max_memory_allocated_mb": torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else None,
        "git": git_info(),
    }

    with open(os.path.join(out_dir, "per_env.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[INFO] success {summary['success']['rate']:.3f} ± {summary['success']['se']:.3f}  "
          f"fail {summary['fail']['rate']:.3f}  stuck {summary['stuck']['rate']:.3f}  "
          f"(n={n}, distinct tiles={summary['tiles_distinct_used']}, truncated={truncated})")
    print(f"[INFO] Results saved to: {out_dir}")
    if args_cli.viz and simulation_app.is_running():
        print("[VIZ] All outcomes decided. The scene is frozen; close the Isaac Sim window to exit.", flush=True)
        while simulation_app.is_running():
            base_env.sim.render()  # renders without stepping physics
            time.sleep(dt)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
