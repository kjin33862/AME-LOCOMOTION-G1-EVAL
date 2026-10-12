# AME evaluation tools for Unitree G1 (Isaac Lab)

Evaluation scripts for the Unitree G1 reproduction of AME
("Attention-Based Map Encoding for Learning Generalized Legged Locomotion", He et al., *Science Robotics* 2025).

The base code, the AME encoder and the pretrained checkpoints (`pretrained/ame1.pt`, `pretrained/ame2.pt`) come from
[SII-FUSC/AME_Locomotion](https://github.com/SII-FUSC/AME_Locomotion). Please see that repository for installation
and training. The upstream repository only plays back videos, so I added a way to measure success rates.

## What is added

- `scripts/rsl_rl/evaluate.py`: runs one checkpoint on a grid of terrain tiles (one robot per tile, 1.0 m/s forward
  command) and counts success / fail / stuck for each robot. `--viz` opens the simulator window for a visual check.
- `scripts/rsl_rl/run_noise_sweep.sh`, `scripts/rsl_rl/aggregate_sweep.py`: the noise sweep and the summary plots.
- Evaluation-only config and MDP terms: `29dof/eval_env_cfg_29dof.py`, `mdp/eval_observations.py`,
  `mdp/eval_terminations.py`.

The training code is not changed. How success and failure are decided, and the known limitations, are described
in [docs/EVALUATION.md](docs/EVALUATION.md).

## Usage

Requires the same setup as upstream (Isaac Sim 5.1.0, Isaac Lab 2.3.0).

```bash
# 225 robots on 15 x 15 stepping-stone tiles, no window
OMNI_KIT_ACCEPT_EULA=YES python -u scripts/rsl_rl/evaluate.py \
    --arch ame1 --terrain steppingstones --num_rows 15 --num_cols 15 --num_envs 225 --seed 0

# 9 robots with the simulator window, 4x slower
OMNI_KIT_ACCEPT_EULA=YES python -u scripts/rsl_rl/evaluate.py \
    --arch ame1 --terrain steppingstones --seed 0 --viz --viz_speed 0.25
```

Each run writes `per_env.csv` (one row per robot) and `summary.json` to `eval_results/`.

## Credits

- G1 reproduction code: [SII-FUSC/AME_Locomotion](https://github.com/SII-FUSC/AME_Locomotion)
- AME: J. He, C. Zhang, F. Jenelten, R. Grandia, M. Bächer, M. Hutter, *Science Robotics* 2025
- AME-2: C. Zhang, V. Klemm, F. Yang, M. Hutter, [arXiv:2601.08485](https://arxiv.org/abs/2601.08485)
- RSL-RL: ETH Zurich and NVIDIA (BSD-3-Clause, see `rsl_rl/LICENSE`)
