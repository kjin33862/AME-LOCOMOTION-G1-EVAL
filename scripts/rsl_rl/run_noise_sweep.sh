#!/usr/bin/env bash
# Overnight noise-robustness sweep of the pretrained AME policies on stepping stones.
#
# Every run: 15x15 tiles, 225 envs, proprio noise on (except C0), one map-noise parameter changed at a time
# from the no-map-noise baseline C1. Output folders end in "_sweep-<condition>-s<seed>"; per-run stdout goes to
# eval_results/sweep_logs/. Restart-safe: a run whose folder already has a summary.json with "valid": true is
# skipped. A failed or invalid run is logged and the sweep moves on.
#
# Launch (from the repo root):
#   nohup systemd-inhibit --what=sleep:idle --why="AME sweep" bash scripts/rsl_rl/run_noise_sweep.sh \
#       > eval_results/sweep_master.log 2>&1 &
# Optional: P2_SEEDS="0 1" to drop seed 2 from Priority 2.

set -u
cd "$(dirname "$0")/../.." || exit 1

PY="$HOME/IsaacLab-2.3.0/env_isaaclab/bin/python"
OUT=eval_results
LOGDIR="$OUT/sweep_logs"
MODELS="ame1 ame2"
P1_SEEDS="0 1 2"
P2_SEEDS="${P2_SEEDS:-0 1 2}"
RUN_TIMEOUT=1800  # [s] kill a hung run after 30 min
FIXED=(--terrain steppingstones --num_rows 15 --num_cols 15 --num_envs 225)
PROPRIO_ON=(--proprio_noise on)
MAP_OFF=(--height_sigma 0 --height_offset 0)

mkdir -p "$LOGDIR"
n_ok=0; n_skip=0; n_bad=0
t_sweep=$(date +%s)

log() { echo "[$(date '+%F %T')] $*"; }

# Prints the folder of a valid run for this (model, tag), if any.
valid_run() {
    local model=$1 tag=$2 f
    for f in "$OUT"/*_"${model}"_steppingstones_*_"${tag}"/summary.json; do
        [ -f "$f" ] || continue
        if "$PY" -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get("valid") is True else 1)' "$f"; then
            dirname "$f"; return 0
        fi
    done
    return 1
}

# run <condition> <model> <seed> [evaluate.py noise args...]
run() {
    local cond=$1 model=$2 seed=$3; shift 3
    local tag="sweep-${cond}-s${seed}"
    local done_dir
    if done_dir=$(valid_run "$model" "$tag"); then
        log "SKIP  $model $tag (valid: $done_dir)"; n_skip=$((n_skip + 1)); return
    fi
    local logf="$LOGDIR/${model}_${tag}_$(date +%Y%m%d-%H%M%S).log"
    log "START $model $tag  args: $*"
    local t0=$(date +%s)
    OMNI_KIT_ACCEPT_EULA=YES timeout --kill-after=60 "$RUN_TIMEOUT" \
        "$PY" -u scripts/rsl_rl/evaluate.py --arch "$model" "${FIXED[@]}" --seed "$seed" --tag "$tag" "$@" \
        > "$logf" 2>&1
    local rc=$? dt=$(( $(date +%s) - t0 ))
    if grep -q "cooking failure" "$logf"; then
        log "WARN  $model $tag: 'cooking failure' in $logf"
    fi
    if done_dir=$(valid_run "$model" "$tag"); then
        log "OK    $model $tag  (${dt}s) $(grep -m1 '\[INFO\] success' "$logf")"
        n_ok=$((n_ok + 1))
    else
        log "FAIL  $model $tag  (${dt}s, exit $rc) $(grep -m1 'INVALID RUN' "$logf") -> see $logf"
        n_bad=$((n_bad + 1))
    fi
}

log "Sweep start. P1 seeds: $P1_SEEDS  P2 seeds: $P2_SEEDS"

# ---- Priority 1: all models x all seeds
for seed in $P1_SEEDS; do
    for model in $MODELS; do
        run C0 "$model" "$seed" --noise off --proprio_noise off
        run C1 "$model" "$seed" "${PROPRIO_ON[@]}" "${MAP_OFF[@]}"
        run C2 "$model" "$seed" "${PROPRIO_ON[@]}"
        run C3 "$model" "$seed" "${PROPRIO_ON[@]}" --xy_shift 0.03
    done
done

# ---- Priority 2: one parameter at a time from C1 (sigma 0, offset 0), ordered by seed
for seed in $P2_SEEDS; do
    for model in $MODELS; do
        for v in 0.06 0.10; do
            run "sigma$v" "$model" "$seed" "${PROPRIO_ON[@]}" --height_sigma "$v" --height_offset 0
        done
        for v in 0.05 0.10 0.15; do
            run "offset$v" "$model" "$seed" "${PROPRIO_ON[@]}" --height_sigma 0 --height_offset "$v"
        done
        for v in 0.03 0.06 0.10; do
            run "xy$v" "$model" "$seed" "${PROPRIO_ON[@]}" "${MAP_OFF[@]}" --xy_shift "$v"
        done
        for v in 5 10; do
            run "delay$v" "$model" "$seed" "${PROPRIO_ON[@]}" "${MAP_OFF[@]}" --map_delay_steps "$v"
        done
    done
done

log "Sweep done in $(( ($(date +%s) - t_sweep) / 60 )) min: ok=$n_ok skipped=$n_skip failed/invalid=$n_bad"
