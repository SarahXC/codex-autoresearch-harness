#!/bin/bash
# ==============================================================================
# launch_ab.sh — Run a sequential A/B comparison of two Codex models
# ==============================================================================
#
# Runs model A for N hours, then model B for N hours, on the same GPU.
# Sequential because both models need full GPU access for training.
#
# Usage:
#   ./launch_ab.sh [hours_per_model] [model_a] [model_b] [tag_a] [tag_b]
#
# Defaults:
#   hours_per_model = 6
#   model_a         = gpt-5.4
#   model_b         = gpt-5.3-codex  (Spark / Cerebras)
#   tag_a           = gpt54
#   tag_b           = spark
#
# Examples:
#   ./launch_ab.sh                        # 6h per model, default models
#   ./launch_ab.sh 4                      # 4h per model
#   ./launch_ab.sh 6 gpt-5.4 o3 gpt54 o3test   # compare 5.4 vs o3
#
# Run in tmux so it survives SSH disconnects:
#   tmux new-session -d -s autoresearch './launch_ab.sh 6'
#   tmux attach -t autoresearch           # watch live
# ==============================================================================

set -euo pipefail

HOURS="${1:-6}"
MODEL_A="${2:-gpt-5.4}"
MODEL_B="${3:-gpt-5.3-codex}"
TAG_A="${4:-gpt54}"
TAG_B="${5:-spark}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "ERROR: OPENAI_API_KEY is not set."
    exit 1
fi

echo "============================================"
echo "  Autoresearch A/B Test"
echo "  Model A: ${MODEL_A} (tag: ${TAG_A}) — ${HOURS}h"
echo "  Model B: ${MODEL_B} (tag: ${TAG_B}) — ${HOURS}h"
echo "  Total runtime: $((HOURS * 2))h"
echo "============================================"
echo ""

# --- Model A ---
echo ">>> Starting Model A: ${MODEL_A} at $(date)"
"$SCRIPT_DIR/run_experiment.sh" "$MODEL_A" "$TAG_A" "$HOURS"
echo ">>> Model A finished at $(date)"
echo ""

# --- Model B ---
echo ">>> Starting Model B: ${MODEL_B} at $(date)"
"$SCRIPT_DIR/run_experiment.sh" "$MODEL_B" "$TAG_B" "$HOURS"
echo ">>> Model B finished at $(date)"
echo ""

# --- Comparison ---
echo ""
echo "============================================"
echo "  COMPARISON SUMMARY"
echo "============================================"
echo ""
echo "--- ${MODEL_A} (${TAG_A}) ---"
if [ -f "$HOME/autoresearch_${TAG_A}/timing.log" ]; then
    grep "FINAL RESULTS" -A 100 "$HOME/autoresearch_${TAG_A}/timing.log"
fi
echo ""
echo "--- ${MODEL_B} (${TAG_B}) ---"
if [ -f "$HOME/autoresearch_${TAG_B}/timing.log" ]; then
    grep "FINAL RESULTS" -A 100 "$HOME/autoresearch_${TAG_B}/timing.log"
fi
