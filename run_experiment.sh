#!/bin/bash
# ==============================================================================
# run_experiment.sh — Codex autoresearch loop for a single model
# ==============================================================================
#
# Wraps `codex exec` in a bash while-loop so the agent runs experiments
# continuously without human intervention. Each iteration:
#   1. Calls codex exec with a prompt to run ONE experiment
#   2. Codex reads the repo, proposes a change, trains, evaluates, keeps/discards
#   3. The wrapper logs timing + metrics, then loops back to step 1
#
# The key insight: codex exec is stateless and exits after each call.
# The loop lives in bash, not in the agent. State is persisted through
# git history and results.tsv, which the agent reads on each invocation.
#
# Usage:
#   ./run_experiment.sh <model> <tag> [max_hours]
#
# Examples:
#   ./run_experiment.sh gpt-5.4 gpt54 6
#   ./run_experiment.sh gpt-5.3-codex spark 6
#   ./run_experiment.sh o3 o3test 4
#
# Outputs:
#   ~/autoresearch_<tag>/results.tsv  — experiment results
#   ~/autoresearch_<tag>/timing.log   — per-iteration timing
#   ~/autoresearch_<tag>/output.log   — full codex output
# ==============================================================================

set -euo pipefail

MODEL="${1:?Usage: $0 <model> <tag> [max_hours]}"
TAG="${2:?Usage: $0 <model> <tag> [max_hours]}"
MAX_HOURS="${3:-6}"
MAX_SECONDS=$((MAX_HOURS * 3600))
SOURCE_DIR="$(cd "$(dirname "$0")" && pwd)"
WORK_DIR="$HOME/autoresearch_${TAG}"
BRANCH="autoresearch/${TAG}"

# ---------------------------------------------------------------------------
# API key: reads from environment. Set OPENAI_API_KEY before running.
# ---------------------------------------------------------------------------
if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "ERROR: OPENAI_API_KEY is not set. Export it before running."
    echo "  export OPENAI_API_KEY='sk-...'"
    exit 1
fi

# ---------------------------------------------------------------------------
# Create an isolated working copy so multiple models can run sequentially
# without stepping on each other's git state.
# ---------------------------------------------------------------------------
if [ ! -d "$WORK_DIR" ]; then
    echo "Creating working copy at $WORK_DIR ..."
    # Copy the autoresearch repo (the one with train.py, program.md, etc.)
    AUTORESEARCH_DIR="${AUTORESEARCH_REPO:-$SOURCE_DIR/autoresearch}"
    if [ ! -d "$AUTORESEARCH_DIR" ]; then
        echo "ERROR: Cannot find autoresearch repo at $AUTORESEARCH_DIR"
        echo "Either clone it there or set AUTORESEARCH_REPO=/path/to/autoresearch"
        exit 1
    fi
    cp -r "$AUTORESEARCH_DIR" "$WORK_DIR"
fi

cd "$WORK_DIR"

# Create a dedicated git branch for this run
if ! git show-ref --verify --quiet "refs/heads/${BRANCH}"; then
    git checkout -b "$BRANCH"
    echo "Created branch $BRANCH"
else
    git checkout "$BRANCH"
    echo "Switched to existing branch $BRANCH"
fi

TIMING_LOG="${WORK_DIR}/timing.log"

echo "========================================" | tee -a "$TIMING_LOG"
echo "Autoresearch Run - $(date -Iseconds)"    | tee -a "$TIMING_LOG"
echo "Model:     $MODEL"                        | tee -a "$TIMING_LOG"
echo "Tag:       $TAG"                           | tee -a "$TIMING_LOG"
echo "Branch:    $BRANCH"                        | tee -a "$TIMING_LOG"
echo "WorkDir:   $WORK_DIR"                      | tee -a "$TIMING_LOG"
echo "Max hours: $MAX_HOURS"                     | tee -a "$TIMING_LOG"
echo "========================================" | tee -a "$TIMING_LOG"

RUN_START=$(date +%s)
ITER=0

while true; do
    # Check time limit
    NOW=$(date +%s)
    ELAPSED_TOTAL=$((NOW - RUN_START))
    if [ "$ELAPSED_TOTAL" -ge "$MAX_SECONDS" ]; then
        echo "[${TAG}] Time limit reached (${MAX_HOURS}h). Stopping after ${ITER} iterations." | tee -a "$TIMING_LOG"
        break
    fi

    ITER=$((ITER + 1))
    ITER_START=$(date +%s)
    REMAINING=$(( (MAX_SECONDS - ELAPSED_TOTAL) / 60 ))
    echo "[${TAG}] iter=${ITER} start=$(date -Iseconds) remaining=${REMAINING}min model=${MODEL}" | tee -a "$TIMING_LOG"

    # -----------------------------------------------------------------------
    # The prompt tells codex to run exactly ONE experiment cycle.
    # It reads program.md for full context, checks results.tsv for history,
    # proposes a change, trains, evaluates, and keeps or discards.
    # -----------------------------------------------------------------------
    PROMPT="You are an autonomous ML researcher. Read program.md fully for context, then follow the experiment loop.

Key rules:
- You can ONLY modify train.py (never modify prepare.py)
- Run training: uv run train.py > run.log 2>&1
- After training, run: grep \"^val_bpb:\|^peak_vram_mb:\" run.log
- If grep output is empty, the run crashed — run: tail -n 50 run.log
- Log ALL results to results.tsv (tab-separated). Do NOT git-commit results.tsv
- If val_bpb improved (lower than best so far), keep the git commit
- If val_bpb is equal or worse, run: git reset --hard HEAD~1
- On iteration 1, establish baseline by running train.py unmodified

This is iteration ${ITER}. Run ONE complete experiment cycle:
1. Check results.tsv for current best val_bpb
2. If no baseline exists, run train.py as-is to establish one
3. Otherwise, propose a single change to train.py, git commit it
4. Train and evaluate
5. Log to results.tsv
6. Keep or discard based on val_bpb
Then stop — the wrapper calls you again for the next iteration."

    # -----------------------------------------------------------------------
    # --dangerously-bypass-approvals-and-sandbox is required because:
    #   1. The default sandbox blocks GPU access (CUDA init fails)
    #   2. The sandbox blocks writes to ~/.cache/uv (uv package manager)
    # This is safe on a dedicated VM with nothing else running.
    # -----------------------------------------------------------------------
    codex exec \
        -m "$MODEL" \
        --dangerously-bypass-approvals-and-sandbox \
        "$PROMPT" \
        2>&1 | tee -a "output.log" || true

    ITER_END=$(date +%s)
    ITER_ELAPSED=$((ITER_END - ITER_START))

    # Grab latest result from results.tsv
    LATEST_BPB="N/A"
    TOTAL_ROWS=0
    if [ -f results.tsv ]; then
        TOTAL_ROWS=$(( $(wc -l < results.tsv) - 1 ))  # subtract header
        LATEST_BPB=$(tail -1 results.tsv | cut -f2)
    fi

    echo "[${TAG}] iter=${ITER} elapsed=${ITER_ELAPSED}s experiments=${TOTAL_ROWS} latest_bpb=${LATEST_BPB}" | tee -a "$TIMING_LOG"
    echo "---" >> "$TIMING_LOG"
done

# ---------------------------------------------------------------------------
# Print final summary
# ---------------------------------------------------------------------------
echo ""
echo "========== FINAL RESULTS ==========" | tee -a "$TIMING_LOG"
echo "Model: $MODEL"                        | tee -a "$TIMING_LOG"
echo "Total iterations: $ITER"              | tee -a "$TIMING_LOG"
echo "Total time: $(($(date +%s) - RUN_START))s" | tee -a "$TIMING_LOG"
if [ -f results.tsv ]; then
    echo "" | tee -a "$TIMING_LOG"
    echo "Results:" | tee -a "$TIMING_LOG"
    cat results.tsv | tee -a "$TIMING_LOG"
fi
echo "====================================" | tee -a "$TIMING_LOG"
