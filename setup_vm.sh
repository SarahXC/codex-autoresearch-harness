#!/bin/bash
# ==============================================================================
# setup_vm.sh — One-shot setup for a fresh GPU VM
# ==============================================================================
#
# Run this on a fresh Ubuntu 22.04 GPU VM to install everything needed
# for autoresearch + the Codex harness.
#
# Usage:
#   curl -sSL https://raw.githubusercontent.com/YOU/codex-autoresearch-harness/main/setup_vm.sh | bash
#
# Or after cloning:
#   ./setup_vm.sh
#
# After running, you still need to:
#   1. Set your OpenAI API key:
#      export OPENAI_API_KEY="sk-proj-..."
#      echo "$OPENAI_API_KEY" | codex login --with-api-key
#   2. Run the experiment:
#      cd ~/codex-autoresearch-harness && ./launch_ab.sh 6
# ==============================================================================

set -euo pipefail

echo "=== Autoresearch VM Setup ==="
echo ""

# --- Node.js 22 ---
echo "[1/6] Installing Node.js 22..."
if ! command -v node &>/dev/null || [[ "$(node --version)" != v22* ]]; then
    curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
    sudo apt-get install -y nodejs
else
    echo "  Node.js $(node --version) already installed"
fi

# --- Codex CLI ---
echo "[2/6] Installing Codex CLI..."
if ! command -v codex &>/dev/null; then
    sudo npm install -g @openai/codex
else
    echo "  Codex $(codex --version) already installed"
fi

# --- uv (Python package manager) ---
echo "[3/6] Installing uv..."
if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
else
    echo "  uv $(uv --version) already installed"
fi

# --- tmux ---
echo "[4/6] Installing tmux..."
sudo apt-get install -y tmux 2>/dev/null || true

# --- Clone autoresearch ---
echo "[5/6] Cloning autoresearch..."
if [ ! -d "$HOME/autoresearch" ]; then
    cd ~
    git clone https://github.com/karpathy/autoresearch.git
    cd autoresearch
    uv sync
    echo "  Running prepare.py to download data..."
    uv run prepare.py
else
    echo "  ~/autoresearch already exists"
fi

# --- Verify ---
echo "[6/6] Verifying setup..."
echo ""
echo "  Node.js:  $(node --version 2>/dev/null || echo 'MISSING')"
echo "  npm:      $(npm --version 2>/dev/null || echo 'MISSING')"
echo "  Codex:    $(codex --version 2>/dev/null || echo 'MISSING')"
echo "  uv:       $(uv --version 2>/dev/null || echo 'MISSING')"
echo "  Python:   $(python3 --version 2>/dev/null || echo 'MISSING')"
echo "  tmux:     $(tmux -V 2>/dev/null || echo 'MISSING')"
echo "  GPU:      $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'NONE')"
echo ""

# Check data
if [ -d "$HOME/.cache/autoresearch/data" ]; then
    SHARDS=$(ls ~/.cache/autoresearch/data/shard_*.parquet 2>/dev/null | wc -l)
    echo "  Data shards: $SHARDS"
else
    echo "  WARNING: No data found. Run: cd ~/autoresearch && uv run prepare.py"
fi

if [ -f "$HOME/.cache/autoresearch/tokenizer/tokenizer.pkl" ]; then
    echo "  Tokenizer:  OK"
else
    echo "  WARNING: No tokenizer found."
fi

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. Export your OpenAI API key:"
echo "     export OPENAI_API_KEY='sk-proj-...'"
echo "     echo \"\$OPENAI_API_KEY\" | codex login --with-api-key"
echo ""
echo "  2. Quick test (verify GPU + training works):"
echo "     cd ~/autoresearch && uv run train.py"
echo ""
echo "  3. Run the A/B experiment:"
echo "     export AUTORESEARCH_REPO=~/autoresearch"
echo "     tmux new-session -s autoresearch"
echo "     cd ~/codex-autoresearch-harness && ./launch_ab.sh 6"
echo ""
