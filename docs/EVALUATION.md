# Quantitative evaluation of AME checkpoints

This fork adds a headless evaluation pipeline for the pretrained AME policies (`pretrained/ame1.pt`,
`pretrained/ame2.pt`). It answers one question per run: *on a grid of terrain tiles, what fraction of robots
walk 4 m before falling?* It also runs a noise-robustness sweep and plots the results.

Nothing here changes training. The training task, rewards, observations and `play.py` are untouched; evaluation
uses its own env config class and its own observation term.

## 1. Files

| file | what it does |
|---|---|
| `source/ame_locomotion/ame_locomotion/tasks/manager_based/ame_locomotion/29dof/eval_env_cfg_29dof.py` | Evaluation env config `G1EvalEnvCfg` (subclass of `G1RoughEnvCfg_PLAY`): one terrain type at a fixed difficulty on a `num_rows x num_cols` grid, env *i* spawns at the center of its own tile (`EvalTerrainImporter`), no camera, no terrain curriculum, 20 s episodes, configurable map and proprio noise. `configure_eval()` applies the CLI settings. |
| `source/ame_locomotion/ame_locomotion/tasks/manager_based/ame_locomotion/mdp/eval_observations.py` | `ElevationMapEval`: the same height-map observation as `observations.elevation_map` (same geometry and clipping) with configurable Gaussian height noise, per-episode height offset, per-episode x-y shift and update delay. Offset and shift are sampled in `reset()`, so they also apply in the first episode. |
| `source/ame_locomotion/ame_locomotion/tasks/manager_based/ame_locomotion/mdp/eval_terminations.py` | `illegal_contact_recorded`: returns exactly what Isaac Lab's `mdp.illegal_contact` returns and also stores which body had the largest contact force when it fired. `G1EvalEnvCfg` uses it for the `base_contact` termination (`eval_env_cfg_29dof.py:71`). |
| `scripts/rsl_rl/evaluate.py` | Runs one evaluation (one checkpoint, one terrain, one noise setting, one seed) and writes `per_env.csv`, `summary.json`, `args.json`. With `--viz`, runs a small grid in the Isaac Sim window for visual checking (section 9). |
| `scripts/rsl_rl/run_noise_sweep.sh` | Runs the stepping-stone noise sweep: 2 models x 14 conditions x 3 seeds = 84 runs of 225 envs. Restart-safe. |
| `scripts/rsl_rl/aggregate_sweep.py` | CPU only. Reads all valid sweep runs, pools seeds, writes `eval_results/sweep_summary.csv` and `eval_results/sweep_plots/*.png`. |

Related changes to existing files (additions only, used by the evaluation):

- `29dof/velocity_env_cfg_29dof.py`: `PLAY_SUB_TERRAINS` (stairs_down, stairs_up, steppingstones tiles) and
  `G1RoughEnvCfg_PLAY_{STAIRS_DOWN,STAIRS_UP,STEPPINGSTONES}`.
- `agents/ame_rsl_rl_ppo_cfg.py`: `G1AMEGlobalPPORunnerCfg` (`attach_global=True`, needed to load `ame2.pt`).
- `29dof/__init__.py`: Play task IDs for the variants above (for `play.py`; `evaluate.py` does not use gym IDs).

## 2. What `evaluate.py` does, step by step

All references are to `scripts/rsl_rl/evaluate.py` unless noted.

1. **Parse the CLI** (`:33-79`; the grid defaults depend on `--viz`), run headless unless `--viz`, with no
   cameras (`:80-82`), then start Isaac Sim
   (`:84-85`). Isaac Lab modules are imported only after the app is running (`:89-114`).
2. **Resolve the noise settings** (`resolve_noise`, `:124-136`). Start from the preset (`stage2` or `off`), then
   apply `--height_sigma` / `--height_offset` if given. `--xy_shift` and `--map_delay_steps` always come from the
   CLI. Proprio noise follows the preset unless `--proprio_noise on|off`.
3. **Build the env config** (`:203-209`): `G1EvalEnvCfg()` then `configure_eval()`
   (`eval_env_cfg_29dof.py:74-97`), which selects the terrain tile type, the grid size, a fixed difficulty
   (`difficulty_range = (d, d)`), the terrain generator seed (= `--seed`), the episode length and the noise.
4. **Build the agent config** (`:210-212`): `G1AMEGlobalPPORunnerCfg` for `ame2`, `G1AMEPPORunnerCfg` for `ame1`.
5. **Create the env and load the policy** (`:215-220`), the same way `play.py` does. The checkpoint's training
   iteration is read for the summary.
6. **Allocate per-env buffers** (`:229-248`). The spawn position is recorded right after the initial reset
   (`:230`). Derived constants: `hold_steps = round(fall_hold_s / step_dt)` = 10 steps with the defaults
   (`step_dt` = 0.005 s x decimation 4 = 0.02 s), `cos_tilt = cos(tilt_deg)`, and the step limit =
   episode length in steps + 5 (1005), or `--max_steps`. With `--viz` it also prints the env -> tile map,
   points the viewport camera at the grid and creates the outcome markers (`:250-263`).
7. **Create the output folder** `<out_root>/<YYYYmmdd-HHMMSS>_<checkpoint name>_<terrain>_<noise tag>[_<tag>]` and
   write `args.json` (`:266-273`). This happens before the rollout so that an aborted run still leaves a folder.
   The noise tag is `noiseoff` or `h<sigma>-o<offset>-xy<shift>-d<delay>-p<proprio 0|1>` (`noise_tag`, `:139-144`).
8. **Set up the validity guard** (`:276-323`, see section 5).
9. **Roll out** (`:325-412`). Each iteration: policy -> `env.step` -> update outcomes (section 3). The loop stops
   when every env has an outcome, or at the step limit. Every 100 steps it prints a progress line. At step 1 and
   at step 50 (= 1 s) it runs the validity guard (`:388-389`). With `--viz`, each step also prints the envs that
   just got an outcome, moves the markers and sleeps to keep real-time speed (`:391-408`).
10. **Finish** (`:414-417`): `truncated_by_max_steps` = some env still had no outcome; run the guard one last time;
    envs still without an outcome become `stuck`.
11. **Build the per-env table** (`:420-449`) and the **summary** (`:452-492`), write `per_env.csv` and
    `summary.json` (`:494-499`), and print the success / fail / stuck rates (`:500-503`). With `--viz`, the window
    then stays open with the scene frozen (rendering without physics) until it is closed (`:504-508`).

Only the **first episode** of each env counts. Isaac Lab resets an env automatically when its episode ends; from
then on the env is ignored (`active` is false).

## 3. How outcomes are decided

Every policy step (20 ms), for envs that are still active, in this order (`:337-386`):

| order | check | result |
|---|---|---|
| 1 | The episode ended this step and it was **not** a time-out (`:340-343`) | `fail`, reason `contact` |
| 1 | The episode ended this step by **time-out** (20 s) (`:344`) | `stuck` |
| 2 | Base x-y distance from spawn >= `--success_dist` (4.0 m) (`:370-373`) | `success` |
| 3 | Tilt condition held for `hold_steps` consecutive steps (`:375-376`, `:381-386`) | `fail`, reason `tilt` |
| 3 | Low-height condition held for `hold_steps` consecutive steps (`:377`) | `fail`, reason `height` |
| 3 | Spawn-drop condition held for `hold_steps` consecutive steps (only with `--spawn_drop_rule`) (`:378-380`) | `fail`, reason `spawn_drop` |
| end | No outcome when the loop stops (only possible with `--max_steps`) (`:416`) | `stuck` |

Details and exact thresholds:

- **contact**: the env's `base_contact` termination (`velocity_env_cfg_29dof.py:400-416`, Isaac Lab
  `mdp.illegal_contact`, threshold **1.0 N**) on any of: `torso_link`, `.*_shoulder_.*_link`, `.*_hip_.*_link`,
  `.*_knee_link`, `.*_elbow_link`, `waist_.*_link`, `pelvis`. Any termination that is not a time-out is counted as
  `contact`, because this is the only other termination term. The touching body is recorded inside the
  termination by `illegal_contact_recorded` (`mdp/eval_terminations.py`), because the automatic reset inside
  `env.step()` zeroes the contact-force history before `evaluate.py` could read it (`:343`). If several bodies
  are above 1 N in that step, the one with the largest force is reported (`contact_body`).
- **Body indices: contact sensor != articulation.** The termination's `body_ids` and `contact_body` use
  **contact-sensor** indices. The contact sensor orders bodies limb by limb (`1:left_hip_pitch_link` ...
  `10:right_knee_link`), while the articulation (`robot.body_names`) interleaves left and right
  (`1:left_hip_pitch_link 2:right_hip_pitch_link` ...). Always convert a sensor index with
  `scene["contact_forces"].body_names`, never with `robot.body_names`; mixing them up mislabels bodies, including
  left and right. Checked on 2026-10-11 (ame1 C2 seed 0, 328 contacts): at contact, every `left_*` body was on the
  robot's left side and every `right_*` body on its right side.
- **success**: Euclidean x-y distance of the root (pelvis) from its spawn point `>= 4.0 m`. Direction is not
  checked. The tiles are 8 x 8 m, so 4 m from the center is the tile edge.
- **tilt**: `-projected_gravity_b.z < cos(60 deg) = 0.5`, i.e. the base is tilted more than **60 deg**.
- **height**: root height above the supporting surface `< 0.4 m`. The supporting surface is the **highest
  height-scanner ray hit within 0.3 m** (x-y) of the root (`:360-365`). If no ray hits within that radius, the
  highest hit of the whole scan is used. Non-finite hits are ignored. For reference, a walking robot is at about
  0.73 m.
- **spawn_drop** (off by default): root z `< spawn z - 0.5 m`. Off by default because it would wrongly fire on
  `stairs_down`.
- **hold**: a fall condition must be true for **10 consecutive steps (0.2 s)**. The counter resets to 0 as soon as
  the condition is false. If several fall rules complete in the same step, the first in the order tilt, height,
  spawn_drop is recorded.
- Ties: a contact termination is checked before success, and success before falls. A robot that crosses 4 m
  while tilted counts as a success.
- **Times**: `fail_time_s` and `time_to_success_s` are `step x 0.02 s` at the step the outcome was assigned. For
  tilt/height falls this is the end of the 0.2 s hold, not the start of the fall.

The velocity command is fixed at 1.0 m/s forward, 0 lateral, heading 0 (from `G1RoughEnvCfg_PLAY`).

## 4. Command-line options

`evaluate.py` options (`:33-69`). Isaac Lab's `AppLauncher` options (e.g. `--device`) are also accepted;
`--headless` is set from `--viz` and `--enable_cameras` is forced off (`:80-82`).

| option | default | meaning |
|---|---|---|
| `--arch` | `ame1` | `ame1` (`G1AMEPPORunnerCfg`) or `ame2` (`G1AMEGlobalPPORunnerCfg`, `attach_global=True`). |
| `--checkpoint` | `pretrained/<arch>.pt` | Checkpoint path. |
| `--terrain` | `steppingstones` | `steppingstones`, `stairs_down`, `stairs_up` or `stakes`. |
| `--num_envs` | 100 (9 with `--viz`) | Number of robots. Should equal `num_rows x num_cols` so every robot has its own tile. |
| `--num_rows` | 10 (3 with `--viz`) | Tile grid rows. |
| `--num_cols` | 10 (3 with `--viz`) | Tile grid columns. |
| `--difficulty` | 0.5 | Fixed terrain difficulty for all tiles. |
| `--seed` | 0 | Seed for the terrain generator, the env and the agent. |
| `--episode_s` | 20.0 | Episode length [s]; reaching it without an outcome = `stuck`. |
| `--noise` | `stage2` | Preset. `stage2`: height sigma 0.03 m, offset +/-0.05 m, proprio noise on. `off`: no noise. |
| `--height_sigma` | preset | Override: std of per-cell Gaussian height noise [m], resampled every step. |
| `--height_offset` | preset | Override: per-env, per-episode height offset ~ U(-x, x) [m]. |
| `--xy_shift` | 0.0 | Per-env, per-episode x-y shift ~ U(-x, x) [m] added to the map's x/y channels. |
| `--map_delay_steps` | 1 | Refresh the map every N policy steps (5 = 10 Hz, 10 = 5 Hz). 1 = every step (off). |
| `--proprio_noise` | `preset` | `preset` follows `--noise`; `on` / `off` force it. Stage-2 uniform noise: ang vel +/-0.2, projected gravity +/-0.05, joint pos +/-0.01, joint vel +/-2.0. |
| `--success_dist` | 4.0 | Success distance [m]. |
| `--tilt_deg` | 60.0 | Tilt fail threshold [deg]. |
| `--fall_height` | 0.4 | Min root height above the supporting surface [m]. |
| `--support_radius` | 0.3 | x-y radius around the root for the supporting surface [m]. |
| `--fall_hold_s` | 0.2 | How long a fall condition must hold [s]. |
| `--spawn_drop_rule` | off | Also fail if root z drops more than `--spawn_drop` below spawn. |
| `--spawn_drop` | 0.5 | Drop for the spawn-drop rule [m]. |
| `--max_steps` | none | Stop after N steps (smoke / memory tests). Marks the run `truncated_by_max_steps`. |
| `--out_root` | `<repo>/eval_results` | Parent folder for run folders. |
| `--tag` | `""` (`viz` with `--viz`) | Suffix for the run folder name; stored in `summary.json`. |
| `--viz` | off | Visual check mode: Isaac Sim window, 3 x 3 grid by default, real-time speed, one terminal line per outcome, colored markers (section 9). |
| `--viz_speed` | 1.0 | `--viz` only: playback speed relative to real time (0.25 = 4x slower). It only changes the pause between steps, not the results. |

## 5. Output files

Each run folder contains `args.json` (the full CLI, `argv` included), `summary.json` and, for valid runs,
`per_env.csv`.

### `per_env.csv` (one row per env)

| column | meaning |
|---|---|
| `env_id` | Env index 0 .. num_envs-1. |
| `tile_row`, `tile_col` | Tile of this env: `(env_id // num_cols) % num_rows`, `env_id % num_cols`. |
| `outcome` | `success`, `fail` or `stuck`. |
| `fail_reason` | `contact`, `tilt`, `height`, `spawn_drop`, or empty. |
| `contact_body` | For `contact` fails: the contact-sensor body with the largest force when the termination fired (e.g. `torso_link`, `right_knee_link`); empty otherwise. |
| `fail_time_s` | Time of the fail [s]; `nan` otherwise. |
| `time_to_success_s` | Time of the success [s]; `nan` otherwise. |
| `max_forward_m` | Largest world-x displacement from spawn while active [m]. |
| `max_xy_disp_m` | Largest x-y distance from spawn while active [m]. |
| `mean_abs_vx_err` | Mean of \|v_x - command_x\| over active steps, with v_x in the yaw-aligned base frame [m/s]. `nan` for failed envs. |
| `tracked_steps` | Number of steps the env was active (includes the step of a success or fall, not the contact step). |
| `mean_height_above_support_m` | Mean root height above the supporting surface while active [m]. |
| `static_friction` | Static friction of the robot's shapes after randomization (mean over shapes). |

### `summary.json` (valid run)

| key | meaning |
|---|---|
| `valid` | `true`. |
| `validity_check` | The final guard result: `valid`, `invalid_reason`, `kit_log` (path), `cooking_check` (`ok` / `failed` / `unavailable`), `cooking_lines`, `early_height_fails`, `checked_at_step`. |
| `checkpoint`, `checkpoint_iter` | Checkpoint path and its training iteration. |
| `arch`, `terrain`, `difficulty`, `seed` | Run settings. |
| `num_envs`, `tiles_total`, `tiles_distinct_used` | Envs, tiles in the grid, and distinct tile indices actually used. |
| `height_noise` | `{height_sigma, height_offset, xy_shift, delay_steps}` actually applied. |
| `proprio_noise` | `true` / `false`. |
| `tag` | The `--tag` value. |
| `success`, `fail`, `stuck` | Each `{count, rate, se}` with `se = sqrt(p(1-p)/n)`. |
| `fail_reasons` | Counts for `contact`, `tilt`, `height`, `spawn_drop`. |
| `mean_time_to_success_s` | Mean over successful envs. |
| `mean_max_forward_m` | Mean `max_forward_m` over all envs. |
| `mean_abs_vx_err_survivors`, `n_survivors` | Mean velocity error over non-failed envs, and how many there are. |
| `median_height_above_support_m` | Median of the per-env mean heights (about 0.73 m when walking normally). |
| `steps_simulated`, `episode_steps`, `step_dt` | Steps actually run (the loop stops early once every env has an outcome), episode length in steps, and the step duration [s]. |
| `truncated_by_max_steps` | `true` if `--max_steps` stopped the run before every env had an outcome. |
| `wall_time_s` | Rollout wall time. |
| `torch_max_memory_allocated_mb` | Peak PyTorch GPU memory (Isaac Sim's own memory is not included). |
| `git` | `commit` and `dirty_files` of the repo at run time. |

An **invalid** `summary.json` has `valid: false`, `invalid_reason`, `validity_check`, and the run settings
(`checkpoint`, `arch`, `terrain`, `seed`, `num_envs`, `tiles_total`, `height_noise`, `proprio_noise`, `tag`,
`steps_simulated`), and no result keys.

## 6. Validity guard

Large stepping-stone grids can make PhysX fail to cook the terrain mesh. The robots then fall through the ground
and every env fails by `height`, which looks like a real (terrible) result. The guard (`:276-323`) catches this:

- **Cooking check**: it greps this process's Kit log (path from the carb setting `/log/file`) for
  `"cooking failure"`. Run at step 1, at step 50 and at the end.
- **Early-height check**: from step 50 (1 s) on, if more than **50 %** of envs have failed by `height` within the
  first **1.0 s**, the run is invalid.

On an invalid run the script writes the invalid `summary.json`, prints `[ERROR] INVALID RUN: <reason>` and stops.
**The exit code is 0 anyway** (closing Isaac Sim ends the process), so never trust the exit code.

How to tell if a run is invalid:

1. `summary.json` has `"valid": false` (the definitive check), and there is no `per_env.csv`.
2. The log contains `INVALID RUN` or `cooking failure`.
3. Also check valid runs: `validity_check.cooking_check` should be `"ok"`. `"unavailable"` means the Kit log could
   not be read, so the cooking check did not run, even though `valid` is `true`.
4. Sanity values: `median_height_above_support_m` near 0.73 (about 0.39 means robots fell through),
   `tiles_distinct_used == num_envs`, `truncated_by_max_steps == false`.

## 7. Known limitations

- **`--xy_shift` only changes the coordinate channels.** The shift is added to the x/y values of each map cell,
  but each cell's height is still measured at its true position. A real localization error would also move the
  heights. This mirrors the (commented-out) line in the training code, but it is not a realistic map
  misregistration.
- **`--map_delay_steps` holds the robot-centric map.** Between refreshes, the policy sees the previous map in the
  previous body frame, as if the terrain moved with the robot. A real low-rate map is fixed in the world and would
  be re-projected to the current pose. The delay results are therefore not a direct model of a slow mapping
  pipeline.
- **Only stepping-stone tiles differ from each other.** With the current single-value parameter ranges and a
  fixed difficulty, stairs and stakes tiles are identical; only stepping stones get random stone offsets. On
  stairs/stakes, envs differ only by friction, initial joint velocities and noise, so `n` overstates the number of
  independent terrain samples. `tiles_distinct_used` counts tile indices, not distinct geometry.
- **225-env limit for stepping stones.** 15 x 15 tiles (225 envs) works; 15 x 16 and 16 x 16 fail PhysX mesh
  cooking. The limit is the terrain mesh, not GPU memory (225 envs peak at about 5.8 GB on an 8 GB GPU). Other
  terrain types were not tested.
- **Stage-2 training of the checkpoints is not confirmed.** Noise is only on in the stage-2 training config
  (`FINETUNE=True`). Both checkpoints are at iteration 21000 while `max_iterations = 10000`, which suggests two
  stages, but nothing in the checkpoint records the env config. Stage 2 also has no stepping-stone tiles.
- **Statistics.** SEs and CIs treat envs as independent. Runs with the same seed share the same terrain, so
  comparisons between conditions are paired and the unpaired CIs are conservative.
- **Randomization.** Pushes and mass/CoM randomization are off (from the play config); friction and initial
  joint-velocity randomization are on (seeded).
- **Not perfectly deterministic.** The same command with the same seed usually reproduces every `per_env.csv`
  cell, but not always: on 2026-10-11, 1 of 8 identical headless reruns (ame2, C2, seed 2) diverged from the first
  steps (164 vs 161 successes; every row differed except the tile and friction columns); two further reruns
  matched the saved run exactly. Treat a single run as one sample; compare conditions over several seeds.
- **`contact_body` names one body.** If several bodies cross 1 N in the same step, only the one with the
  largest force is reported.
- **Success does not check direction.** Any x-y direction counts, so a robot drifting sideways could reach 4 m.
  In the 2026-10-11 sweep (84 valid runs, 10,202 successful envs) the smallest `max_forward_m` among successes
  was 3.77 m (28 below 3.9 m, none below 3.5 m), so this did not matter there.
- **Supporting-surface fallback (fix before evaluating stairs).** If no ray hits within `--support_radius` of
  the root, the highest hit of the whole scan is used (`:364`). On stairs that can be a higher step ahead, which
  makes the base look low and can cause false `height` fails.

## 8. Reproducing the results

Prerequisites: Isaac Lab 2.3.0 environment (here `~/IsaacLab-2.3.0/env_isaaclab`), the EULA accepted
non-interactively with `OMNI_KIT_ACCEPT_EULA=YES`, and `python -u` when output goes to a file (otherwise the last
prints can be lost when Isaac Sim closes).

Single run:

```bash
OMNI_KIT_ACCEPT_EULA=YES ~/IsaacLab-2.3.0/env_isaaclab/bin/python -u scripts/rsl_rl/evaluate.py \
    --arch ame1 --terrain steppingstones --num_rows 15 --num_cols 15 --num_envs 225 --seed 0
```

Full sweep (84 runs, about 50 min on an RTX 5060 Laptop GPU), from the repo root:

```bash
nohup systemd-inhibit --what=sleep:idle --why="AME sweep" bash scripts/rsl_rl/run_noise_sweep.sh \
    > eval_results/sweep_master.log 2>&1 &
```

- Conditions: C0 (no noise), C1 (proprio noise only, the baseline), C2 (stage-2 map noise), C3 (C2 + 3 cm x-y
  shift), then one parameter at a time from C1: `sigma0.06/0.10`, `offset0.05/0.10/0.15`, `xy0.03/0.06/0.10`,
  `delay5/10`.
- Run folders end in `_sweep-<condition>-s<seed>`; per-run logs go to `eval_results/sweep_logs/`.
- Restart-safe: a run whose folder already has a valid `summary.json` is skipped. To redo a run, move its folder
  away first.
- `P2_SEEDS="0 1"` drops seed 2 from the one-parameter-at-a-time part.

Table and plots (CPU only, a few seconds):

```bash
~/IsaacLab-2.3.0/env_isaaclab/bin/python scripts/rsl_rl/aggregate_sweep.py
```

It keeps only valid, non-truncated runs whose tag matches `sweep-<cond>-s<seed>` (the latest folder wins if a
run was repeated), pools seeds per (model, condition) as one sample of `n = sum(num_envs)` envs, and reports
rates, SEs and **95 % Wilson score intervals**. Outputs:

- `eval_results/sweep_summary.csv`
- `eval_results/sweep_plots/priority1.png`, `height_sigma.png`, `height_offset.png`, `xy_shift.png`,
  `map_delay_steps.png`

The results of the 2026-10-11 sweep are summarized in `notes/eval_log.md`.

## 9. Visual check mode (`--viz`)

Use this to watch the robots and check the scoring by eye. It is not meant for statistics (9 robots by default).

```bash
OMNI_KIT_ACCEPT_EULA=YES ~/IsaacLab-2.3.0/env_isaaclab/bin/python -u scripts/rsl_rl/evaluate.py \
    --arch ame1 --terrain steppingstones --seed 0 --viz --viz_speed 0.25
```

- `--viz_speed 0.25` plays 4x slower than real time (default 1.0); the results do not depend on it.
- The Isaac Sim window opens; the camera looks at the grid from behind-left. Robots walk toward +x.
- At the start the terminal prints `env i -> tile (row, col)` and each spawn x, y, so you can find each robot.
- Each robot has a sphere 0.8 m above its base: **white** running, **green** success, **red** fail, **yellow** stuck.
- The moment a robot gets its outcome, one line is printed, e.g.
  `[t= 3.24 s] env 4 (tile 1,1)  FAIL     reason=contact  body=left_knee_link  x=2.1 m`.
  `x` is the forward (world x) displacement from spawn. For contact fails it is taken from the previous step,
  because the robot has already been reset.
- After a fail the robot reappears standing at its tile center (Isaac Lab's automatic reset). It is ignored from
  then on; its sphere stays where the outcome happened.
- The run uses the same outcome rules as headless and saves its results too (folder tag `viz`). After the last
  outcome the scene is frozen; close the window to exit.
- Headless output is unchanged by `--viz` and `contact_body`: re-running ame1 C2 seed 0 reproduced the saved
  `summary.json` and all original `per_env.csv` columns exactly (2026-10-11).
