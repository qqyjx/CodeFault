#!/usr/bin/env bash
# experiments/run_mf_all.sh
#
# A800 unattended orchestrator for the ASE 2026 #759 Major Revision experiments.
#
# What it does:
#   1. pulls the latest main from origin
#   2. validates data, GPU, python environment
#   3. spawns two detached tmux sessions:
#        - mf_card0_jitsmart : MF1 SimpleJITSmartLog on card 0
#        - mf_card1_codeflow : MF1 SimpleCodeFlowLMLog on card 0 after JITSmart finishes
#        - mf_card1_mf2      : MF2 within-type + control on card 1
#   4. spawns a watchdog tmux session that polls every 10 min, commits results JSON
#      and logs to GitHub when each phase finishes, and writes _DONE markers.
#   5. you SSH in once, run this, and walk away. Final push lands on origin/main.
#
# Re-running this script is safe: it skips phases whose result JSON already exists
# and only restarts whatever did not finish. Use FORCE=1 to re-run everything.
#
# Usage:
#   ssh xyf@zkti
#   cd ~/ASE2026
#   bash experiments/run_mf_all.sh
#   # then disconnect
#
# Force a clean rerun:
#   FORCE=1 bash experiments/run_mf_all.sh

set -uo pipefail

REPO_DIR="${REPO_DIR:-$HOME/ASE2026}"
SEEDS="${SEEDS:-42 1337 2024}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-2e-5}"
LOG_DIR="$REPO_DIR/logs"
RESULTS_DIR="$REPO_DIR/results"
TS="$(date +%Y%m%d_%H%M%S)"

# Hard-code python path to the conda env we bootstrapped, so preflight + tmux
# children all use torch + scikit-learn + drain3 from the right env.
PY_BIN="${PY_BIN:-/data0/xyf/envs/ase2026/bin/python}"

mkdir -p "$LOG_DIR"

cd "$REPO_DIR" || { echo "ERROR: repo dir $REPO_DIR not found"; exit 2; }

# ---------------------------------------------------------------------------
# 0. Preflight
# ---------------------------------------------------------------------------

echo "[$(date)] preflight" | tee -a "$LOG_DIR/orchestrator_$TS.log"

git fetch origin 2>&1 | tee -a "$LOG_DIR/orchestrator_$TS.log"
git checkout main 2>&1 | tee -a "$LOG_DIR/orchestrator_$TS.log"
git pull --ff-only origin main 2>&1 | tee -a "$LOG_DIR/orchestrator_$TS.log"

if [ ! -x "$PY_BIN" ]; then
    echo "ERROR: conda env python not found at $PY_BIN" \
        | tee -a "$LOG_DIR/orchestrator_$TS.log"
    exit 3
fi

"$PY_BIN" -c 'import torch; assert torch.cuda.is_available(), "no CUDA"' \
    || { echo "ERROR: CUDA not available via $PY_BIN"; exit 3; }

n_gpus=$("$PY_BIN" -c 'import torch; print(torch.cuda.device_count())')
echo "  GPUs visible: $n_gpus" | tee -a "$LOG_DIR/orchestrator_$TS.log"
if [ "$n_gpus" -lt 2 ]; then
    echo "WARN: only $n_gpus GPU visible; will serialize phases on card 0" \
        | tee -a "$LOG_DIR/orchestrator_$TS.log"
fi

for f in train.json val.json test.json; do
    if [ ! -s data/processed/codefault/"$f" ]; then
        echo "ERROR: data/processed/codefault/$f missing or empty" \
            | tee -a "$LOG_DIR/orchestrator_$TS.log"
        exit 4
    fi
done
echo "  data OK" | tee -a "$LOG_DIR/orchestrator_$TS.log"

if ! command -v tmux >/dev/null; then
    echo "ERROR: tmux not installed (apt install tmux)" \
        | tee -a "$LOG_DIR/orchestrator_$TS.log"
    exit 5
fi

# ---------------------------------------------------------------------------
# 1. Phase planning. Each phase has: id, gpu, command, output JSON
# ---------------------------------------------------------------------------

# Format: "id|gpu|outfile|extra_args"
PHASES=(
    "mf1_jitsmart|0|$RESULTS_DIR/mf1_strong_fusion_baselines.json|--mode mf1 --seeds $SEEDS"
    "mf2_within|1|$RESULTS_DIR/mf2_within_type_random.json|--mode mf2 --seeds $SEEDS"
)
# Note: run_mf_experiments.py with --mode mf1 already runs BOTH JIT-Smart+log and
# CodeFlowLM+log sequentially on the same card, writing a single JSON. So we only
# need two phases total: MF1 on card 0, MF2 on card 1.

# Optional override: rerun everything if FORCE=1
FORCE="${FORCE:-0}"

# ---------------------------------------------------------------------------
# 2. Spawn phases that are not yet done
# ---------------------------------------------------------------------------

launch_phase() {
    local id="$1"
    local gpu="$2"
    local outfile="$3"
    local extra="$4"
    local logfile="$LOG_DIR/${id}_${TS}.log"
    local sess="mf_${id}"

    if [ "$FORCE" != "1" ] && [ -s "$outfile" ]; then
        echo "  SKIP $id: $outfile already exists (use FORCE=1 to override)" \
            | tee -a "$LOG_DIR/orchestrator_$TS.log"
        return 0
    fi

    if tmux has-session -t "$sess" 2>/dev/null; then
        echo "  SKIP $id: tmux session $sess already running" \
            | tee -a "$LOG_DIR/orchestrator_$TS.log"
        return 0
    fi

    echo "  launching $id (gpu=$gpu) -> $logfile" \
        | tee -a "$LOG_DIR/orchestrator_$TS.log"

    tmux new-session -d -s "$sess" \
        "cd '$REPO_DIR' && \
         CUDA_VISIBLE_DEVICES=$gpu \
         '$PY_BIN' -u experiments/run_mf_experiments.py \
            $extra \
            --epochs $EPOCHS --batch-size $BATCH_SIZE --lr $LR \
            2>&1 | tee '$logfile' ; \
         echo \"[\$(date)] phase $id exited \$?\" >> '$logfile'"
}

for spec in "${PHASES[@]}"; do
    IFS='|' read -r id gpu outfile extra <<< "$spec"
    launch_phase "$id" "$gpu" "$outfile" "$extra"
done

# ---------------------------------------------------------------------------
# 3. Spawn watchdog: polls every 10 min, commits and pushes when each phase
# completes; writes _DONE marker when all phases done.
# ---------------------------------------------------------------------------

WATCHDOG_SESS="mf_watchdog"
WATCHDOG_LOG="$LOG_DIR/watchdog_${TS}.log"
PHASE_SPECS=$(printf "%s\n" "${PHASES[@]}")

if tmux has-session -t "$WATCHDOG_SESS" 2>/dev/null; then
    echo "  watchdog already running, leaving it" \
        | tee -a "$LOG_DIR/orchestrator_$TS.log"
else
    cat > "$REPO_DIR/experiments/_mf_watchdog.sh" <<'WATCH'
#!/usr/bin/env bash
# Watchdog auto-generated by run_mf_all.sh.  Polls every 600s, commits + pushes
# each finished phase JSON, writes _DONE markers, and exits when all phases done.

set -uo pipefail
REPO_DIR="${REPO_DIR:-$HOME/ASE2026}"
LOG_DIR="$REPO_DIR/logs"
RESULTS_DIR="$REPO_DIR/results"
SLEEP_SECS="${SLEEP_SECS:-600}"
MAX_HOURS="${MAX_HOURS:-72}"
DEADLINE=$(( $(date +%s) + MAX_HOURS*3600 ))
cd "$REPO_DIR"

# Encode phase outfile -> tmux session
declare -A SESSION_FOR
SESSION_FOR["$RESULTS_DIR/mf1_strong_fusion_baselines.json"]="mf_mf1_jitsmart"
SESSION_FOR["$RESULTS_DIR/mf2_within_type_random.json"]="mf_mf2_within"

declare -A PUSHED
all_done=0

while [ $all_done -eq 0 ] && [ $(date +%s) -lt $DEADLINE ]; do
    sleep "$SLEEP_SECS"
    all_done=1
    for outfile in "${!SESSION_FOR[@]}"; do
        sess="${SESSION_FOR[$outfile]}"
        if [ -s "$outfile" ] && [ -z "${PUSHED[$outfile]:-}" ]; then
            echo "[$(date)] finished: $outfile"
            git add "$outfile" logs/*.log 2>/dev/null
            git commit -q -m "data($(basename "$outfile" .json)): autopush by watchdog" \
                || echo "  nothing to commit"
            git push origin main 2>&1 | tail -3
            PUSHED["$outfile"]=1
        fi
        if [ -z "${PUSHED[$outfile]:-}" ]; then
            all_done=0
            if ! tmux has-session -t "$sess" 2>/dev/null; then
                echo "[$(date)] WARN: $sess session is gone but $outfile not produced; check log" >&2
                # Do not retry automatically; surface to user via _FAILED marker
                touch "$LOG_DIR/_FAILED_${sess}"
                PUSHED["$outfile"]="failed"
            fi
        fi
    done
done

if [ $(date +%s) -ge $DEADLINE ]; then
    touch "$LOG_DIR/_TIMEOUT"
    echo "[$(date)] watchdog hit MAX_HOURS=$MAX_HOURS limit, exiting"
    exit 6
fi

touch "$LOG_DIR/_DONE"
echo "[$(date)] all phases done"
git add logs/_DONE
git commit -q -m "marker: MF experiments done (watchdog)"
git push origin main 2>&1 | tail -3
WATCH
    chmod +x "$REPO_DIR/experiments/_mf_watchdog.sh"
    tmux new-session -d -s "$WATCHDOG_SESS" \
        "bash $REPO_DIR/experiments/_mf_watchdog.sh 2>&1 | tee '$WATCHDOG_LOG'"
    echo "  watchdog launched -> $WATCHDOG_LOG" | tee -a "$LOG_DIR/orchestrator_$TS.log"
fi

# ---------------------------------------------------------------------------
# 4. Summary
# ---------------------------------------------------------------------------

echo "" | tee -a "$LOG_DIR/orchestrator_$TS.log"
echo "[$(date)] orchestration complete; all sessions are tmux-detached" \
    | tee -a "$LOG_DIR/orchestrator_$TS.log"
echo "" | tee -a "$LOG_DIR/orchestrator_$TS.log"
echo "Current tmux sessions:" | tee -a "$LOG_DIR/orchestrator_$TS.log"
tmux ls | tee -a "$LOG_DIR/orchestrator_$TS.log"
echo "" | tee -a "$LOG_DIR/orchestrator_$TS.log"
echo "Current GPU usage:" | tee -a "$LOG_DIR/orchestrator_$TS.log"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total \
    --format=csv 2>&1 | tee -a "$LOG_DIR/orchestrator_$TS.log"
echo "" | tee -a "$LOG_DIR/orchestrator_$TS.log"
echo "What to watch:"
echo "  tail -f $LOG_DIR/orchestrator_${TS}.log"
echo "  tail -f $LOG_DIR/mf1_jitsmart_${TS}.log"
echo "  tail -f $LOG_DIR/mf2_within_${TS}.log"
echo "  tail -f $LOG_DIR/watchdog_${TS}.log"
echo ""
echo "Healthcheck one-liner:"
echo "  bash $REPO_DIR/experiments/_mf_healthcheck.sh"

# ---------------------------------------------------------------------------
# 5. Write a self-contained healthcheck script
# ---------------------------------------------------------------------------

cat > "$REPO_DIR/experiments/_mf_healthcheck.sh" <<'HEALTH'
#!/usr/bin/env bash
# Quick at-a-glance status. Run any time from the A800.
REPO_DIR="${REPO_DIR:-$HOME/ASE2026}"
cd "$REPO_DIR"
echo "=== $(date) ==="
echo
echo "--- tmux sessions ---"
tmux ls 2>/dev/null || echo "  no tmux sessions"
echo
echo "--- GPU ---"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total \
    --format=csv
echo
echo "--- result files ---"
for f in results/mf1_strong_fusion_baselines.json results/mf2_within_type_random.json; do
    if [ -s "$f" ]; then
        sz=$(stat -c%s "$f")
        mt=$(date -r "$f" +"%Y-%m-%d %H:%M:%S")
        echo "  [done] $f  (size $sz bytes, mtime $mt)"
    else
        echo "  [...] $f  (not yet)"
    fi
done
echo
echo "--- markers ---"
for m in logs/_DONE logs/_TIMEOUT logs/_FAILED_*; do
    [ -e "$m" ] && echo "  $m present" || true
done
echo
echo "--- latest log lines per phase ---"
for f in logs/orchestrator_*.log logs/mf1_jitsmart_*.log logs/mf2_within_*.log logs/watchdog_*.log; do
    if [ -s "$f" ]; then
        echo "=== $f (last 5 lines) ==="
        tail -5 "$f"
        echo
    fi
done
HEALTH
chmod +x "$REPO_DIR/experiments/_mf_healthcheck.sh"

echo ""
echo "Disconnect SSH whenever. The watchdog will commit results JSON and push to"
echo "origin/main as each phase finishes; check back later with healthcheck or"
echo "by pulling on WSL: git pull origin main && cat results/mf1_*.json"
