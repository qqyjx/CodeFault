#!/usr/bin/env bash
# experiments/bootstrap_data.sh
#
# Reconstruct the unified 36K-sample split from public sources, then chain into
# experiments/run_mf_all.sh.
#
# Required because data/processed/codefault/* is .gitignored. Re-creating it
# from scratch reproduces the same dataset construction pipeline described in
# §4.1 of the paper, modulo small split-seed differences from the collaborator's
# original run; the MF2 control variant explicitly re-trains CodeFault on this
# split so the within-type-random vs. control comparison is fair regardless.
#
# Usage on A800 (unattended):
#   ssh xyf@zkti
#   cd ~/ASE2026
#   nohup bash experiments/bootstrap_data.sh > logs/bootstrap_data.log 2>&1 &
#   disown
#
# Idempotent: skips any stage whose output is already present.

set -uo pipefail

REPO_DIR="${REPO_DIR:-$HOME/ASE2026}"
LOG_DIR="$REPO_DIR/logs"
RAW_DIR="$REPO_DIR/data/raw"
PROC_DIR="$REPO_DIR/data/processed/codefault"
ENV_PREFIX="${ENV_PREFIX:-/data0/xyf/envs/ase2026}"

mkdir -p "$LOG_DIR" "$RAW_DIR" "$PROC_DIR"
cd "$REPO_DIR"

ts() { date +"%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(ts)] $*"; }

# ---------------------------------------------------------------------------
# 0. Wait for conda env to be ready (may still be installing in background)
# ---------------------------------------------------------------------------

log "waiting for conda env at $ENV_PREFIX"
for _ in $(seq 1 180); do  # up to 30 min
    if [ -x "$ENV_PREFIX/bin/python" ] && \
       "$ENV_PREFIX/bin/python" -c 'import torch, sklearn, pandas, drain3' 2>/dev/null; then
        log "conda env ready"
        break
    fi
    sleep 10
done
if ! [ -x "$ENV_PREFIX/bin/python" ]; then
    log "ERROR: conda env never came up; aborting"
    exit 2
fi

PY="$ENV_PREFIX/bin/python"

# ---------------------------------------------------------------------------
# 1. Download / clone raw sources (idempotent: skip if dir exists and non-empty)
# ---------------------------------------------------------------------------

clone_if_missing() {
    local url="$1" dest="$2" depth="${3:-1}"
    if [ -d "$dest" ] && [ -n "$(ls -A "$dest" 2>/dev/null)" ]; then
        log "skip clone (exists): $dest"
        return 0
    fi
    log "clone $url -> $dest"
    git clone --depth "$depth" "$url" "$dest" 2>&1 | tail -3
}

clone_if_missing "https://github.com/rjust/defects4j.git" "$RAW_DIR/defects4j"
clone_if_missing "https://github.com/soarsmu/BugsInPy.git" "$RAW_DIR/bugsinpy"
clone_if_missing "https://github.com/logpai/loghub.git" "$RAW_DIR/loghub"

# ManySStuBs4J: Zenodo direct download
MANY_DEST="$RAW_DIR/manysstubs/manysstubs4j.json"
if [ -s "$MANY_DEST" ]; then
    log "skip ManySStuBs4J (exists, $(stat -c%s "$MANY_DEST") bytes)"
else
    log "download ManySStuBs4J from Zenodo"
    mkdir -p "$RAW_DIR/manysstubs"
    curl -L --retry 3 --retry-delay 10 \
        "https://zenodo.org/record/3653444/files/manysstubs4j.json.gz" \
        -o "$RAW_DIR/manysstubs/manysstubs4j.json.gz" 2>&1 | tail -5
    gunzip -kf "$RAW_DIR/manysstubs/manysstubs4j.json.gz"
fi

# ---------------------------------------------------------------------------
# 2. Run each per-source processor (skips silently if outputs exist)
# ---------------------------------------------------------------------------

run_processor() {
    local label="$1" script="$2" outpath="$3"
    if [ -s "$outpath" ]; then
        log "skip $label (exists: $outpath)"
        return 0
    fi
    log "process $label"
    cd "$REPO_DIR/experiments/scripts" || return 1
    "$PY" -u "$script" 2>&1 | tee -a "$LOG_DIR/process_${label}.log"
    cd "$REPO_DIR"
}

run_processor "defects4j" "process_defects4j.py" "$REPO_DIR/data/processed/defects4j.json"
run_processor "bugsinpy" "process_bugsinpy.py" "$REPO_DIR/data/processed/bugsinpy.json"
run_processor "manysstubs" "process_manysstubs.py" "$REPO_DIR/data/processed/manysstubs.json"
run_processor "normal_commits" "mine_normal_commits.py" "$REPO_DIR/data/processed/normal_commits.json"

# ---------------------------------------------------------------------------
# 3. Merge sources -> train/val/test (idempotent: skip if files exist)
# ---------------------------------------------------------------------------

if [ -s "$PROC_DIR/train.json" ] && [ -s "$PROC_DIR/val.json" ] && [ -s "$PROC_DIR/test.json" ]; then
    log "skip merge (final splits already exist)"
else
    log "merge into train/val/test"
    cd "$REPO_DIR/experiments/scripts"
    "$PY" -u merge_datasets.py 2>&1 | tee -a "$LOG_DIR/merge.log"
    cd "$REPO_DIR"
fi

# ---------------------------------------------------------------------------
# 4. Sanity report
# ---------------------------------------------------------------------------

log "final dataset summary"
"$PY" - <<PY
import json, os
for split in ["train", "val", "test"]:
    p = f"$PROC_DIR/{split}.json"
    if not os.path.isfile(p):
        print(f"  MISSING: {p}")
        continue
    data = json.load(open(p))
    pos = sum(1 for d in data if d.get("label", 0) == 1)
    print(f"  {split}: {len(data)} samples, {pos} positive ({100*pos/max(len(data),1):.1f}%)")
PY

if [ -s "$PROC_DIR/train.json" ] && [ -s "$PROC_DIR/val.json" ] && [ -s "$PROC_DIR/test.json" ]; then
    log "DATA READY"
    touch "$LOG_DIR/_DATA_DONE"
    # Hand off to the MF orchestrator
    log "chaining into run_mf_all.sh"
    exec bash "$REPO_DIR/experiments/run_mf_all.sh"
else
    log "ERROR: one or more final splits missing; check $LOG_DIR/merge.log"
    touch "$LOG_DIR/_DATA_FAILED"
    exit 3
fi
