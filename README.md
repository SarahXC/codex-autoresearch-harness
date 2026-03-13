# Codex Autoresearch Harness

A bash harness that makes [OpenAI Codex CLI](https://github.com/openai/codex) run Karpathy's [autoresearch](https://github.com/karpathy/autoresearch) experiments in a continuous loop. Includes an A/B testing framework to compare different models (e.g. GPT-5.4 vs GPT-5.3-Codex Spark on Cerebras).

## Background

Andrej Karpathy's autoresearch gives an LLM agent a neural net training script (`train.py`) and tells it to improve it autonomously — propose changes, train for 5 minutes, evaluate, keep or discard, repeat forever.

The problem: **agents don't want to loop forever.** Karpathy runs this with Claude Code's `/loop` command. Codex CLI has no equivalent — `codex exec` runs once and exits. Our solution is dead simple: wrap `codex exec` in a bash `while true` loop. Each iteration, Codex gets a prompt to run one experiment cycle, reads git history + `results.tsv` to understand what's been tried, and exits. The bash loop calls it again.

```
┌─────────────────────────────────────────────┐
│  bash while loop (run_experiment.sh)        │
│                                             │
│  ┌───────────────────────────────────────┐  │
│  │ codex exec (iteration N)              │  │
│  │  1. Read program.md + results.tsv     │  │
│  │  2. Propose change to train.py        │  │
│  │  3. git commit                        │  │
│  │  4. uv run train.py  (5 min)         │  │
│  │  5. Evaluate val_bpb                  │  │
│  │  6. Keep or git reset --hard HEAD~1   │  │
│  │  7. Log to results.tsv                │  │
│  │  8. Exit                              │  │
│  └───────────────────────────────────────┘  │
│                                             │
│  Loop back → codex exec (iteration N+1)     │
└─────────────────────────────────────────────┘
```

State is persisted between iterations through:
- **Git history** — accepted changes stay as commits; rejected changes are reverted
- **results.tsv** — full experiment log (not committed, stays on disk)
- **program.md** — the agent reads this every iteration for instructions

## What's in this repo

```
codex-autoresearch-harness/
├── README.md              # You are here
├── run_experiment.sh      # Core loop: runs one model for N hours
├── launch_ab.sh           # Orchestrator: runs model A then model B sequentially
├── compare_results.sh     # Post-run analysis: prints side-by-side comparison
└── setup_vm.sh            # One-shot VM setup script (Node, Codex, uv, data)
```

You also need the [autoresearch repo](https://github.com/karpathy/autoresearch) itself (contains `train.py`, `program.md`, `prepare.py`, etc.). This harness wraps around it.

## Prerequisites

- An **H100 (or A100) GPU VM** — we used [Shadeform](https://shadeform.ai) with 1× H100 80GB
- An **OpenAI API key** with access to the models you want to test
- Basic familiarity with tmux and SSH

## Full Setup Guide (from zero to running)

### Step 1: Spin up a GPU VM

We used Shadeform, but any provider works (Lambda, RunPod, Vast.ai, etc.). Requirements:
- **GPU**: 1× H100 80GB (or A100 80GB — will be slower but works)
- **OS**: Ubuntu 22.04
- **Disk**: 100GB+ (for data, venv, model caches)
- **CUDA**: 12.2+ pre-installed

SSH into your VM:
```bash
ssh user@<your-vm-ip>
```

### Step 2: Install system dependencies

```bash
# Install Node.js 22 (required for Codex CLI)
curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
sudo apt-get install -y nodejs

# Install uv (Python package manager — used by autoresearch)
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# Install tmux if not present
sudo apt-get install -y tmux

# Verify
node --version    # v22.x
npm --version     # 10.x
uv --version      # 0.x
nvidia-smi        # Should show your GPU
```

### Step 3: Install and authenticate Codex CLI

```bash
# Install globally
sudo npm install -g @openai/codex

# Verify
codex --version   # 0.114.0 or later

# Authenticate — pipe your API key into codex login
echo "sk-proj-YOUR_KEY_HERE" | codex login --with-api-key

# Also export it (the harness scripts read from env)
echo 'export OPENAI_API_KEY="sk-proj-YOUR_KEY_HERE"' >> ~/.bashrc
source ~/.bashrc
```

**Important**: `codex login --with-api-key` stores the key for the interactive/exec CLI. You also need `OPENAI_API_KEY` in your environment because the harness script checks for it.

### Step 4: Clone and prepare autoresearch

```bash
cd ~

# Fork or clone Karpathy's autoresearch
git clone https://github.com/karpathy/autoresearch.git
cd autoresearch

# Install Python dependencies (creates .venv automatically)
uv sync

# Download and prepare the training data + tokenizer
uv run prepare.py

# Verify data exists
ls ~/.cache/autoresearch/data/       # Should show shard_*.parquet files
ls ~/.cache/autoresearch/tokenizer/  # Should show tokenizer.pkl, token_bytes.pt
```

### Step 5: Verify baseline training works

Before running the harness, confirm that training works on your GPU:

```bash
cd ~/autoresearch
uv run train.py 2>&1 | tail -5
```

You should see output ending with something like:
```
---
val_bpb:          1.044291
training_seconds: 300.1
total_seconds:    336.2
peak_vram_mb:     45060.2
```

If you get CUDA errors, check `nvidia-smi` and ensure your CUDA drivers are compatible with PyTorch 2.9.

### Step 6: Clone this harness

```bash
cd ~
git clone https://github.com/YOUR_USERNAME/codex-autoresearch-harness.git
chmod +x codex-autoresearch-harness/*.sh
```

### Step 7: Run the experiment

#### Option A: A/B comparison (two models, sequential)

```bash
# Start a tmux session (survives SSH disconnects)
tmux new-session -s autoresearch

# Set your API key
export OPENAI_API_KEY="sk-proj-..."

# Point the harness at the autoresearch repo
export AUTORESEARCH_REPO="$HOME/autoresearch"

# Launch: 6 hours per model, 12 hours total
cd ~/codex-autoresearch-harness
./launch_ab.sh 6

# Detach from tmux: Ctrl+B then D
# Reattach later: tmux attach -t autoresearch
```

This runs GPT-5.4 for 6 hours, then GPT-5.3-Codex (Spark/Cerebras) for 6 hours.

To compare different models:
```bash
# Compare GPT-5.4 vs o3
./launch_ab.sh 4 gpt-5.4 o3 gpt54 o3test

# Compare any two models
./launch_ab.sh 6 <model_a> <model_b> <tag_a> <tag_b>
```

#### Option B: Single model run

```bash
export OPENAI_API_KEY="sk-proj-..."
export AUTORESEARCH_REPO="$HOME/autoresearch"
cd ~/codex-autoresearch-harness
./run_experiment.sh gpt-5.4 myrun 8   # Run for 8 hours
```

### Step 8: Monitor progress

While the experiment is running:

```bash
# Watch live output
tmux attach -t autoresearch

# Check timing (from another terminal)
cat ~/autoresearch_gpt54/timing.log | tail -20

# Check experiment results
cat ~/autoresearch_gpt54/results.tsv

# Check GPU utilization
nvidia-smi

# Quick status: latest val_bpb for each run
tail -1 ~/autoresearch_gpt54/results.tsv
tail -1 ~/autoresearch_spark/results.tsv
```

### Step 9: Analyze results

After the run completes:
```bash
cd ~/codex-autoresearch-harness
./compare_results.sh gpt54 spark
```

## How the Codex Loop Works

The core problem: Codex CLI's `codex exec` runs a single task and exits. There is no `/loop` command like Claude Code has. Karpathy himself [confirmed](https://x.com/karpathy/status/1899175870982119875) that "Codex is a known issue :( It basically don't work with autoresearch sadly."

Our workaround:

### The bash wrapper (`run_experiment.sh`)

```bash
while true; do
    # Check time limit
    ...

    # Call codex for ONE experiment
    codex exec \
        -m "$MODEL" \
        --dangerously-bypass-approvals-and-sandbox \
        "$PROMPT" \
        2>&1 | tee -a "output.log" || true

    # Log timing + metrics
    ...
done
```

Each `codex exec` call:
1. Gets a prompt saying "this is iteration N, run one experiment"
2. Reads `program.md` to understand the protocol
3. Reads `results.tsv` to see what's been tried and what the current best is
4. Reads `train.py` (the code it's modifying)
5. Proposes a change, `git commit`s it
6. Runs `uv run train.py > run.log 2>&1` (~5 min training)
7. Checks `val_bpb` from the log
8. If improved: keeps the commit. If not: `git reset --hard HEAD~1`
9. Logs the result to `results.tsv`
10. Exits — bash loop calls it again

### Why `--dangerously-bypass-approvals-and-sandbox`?

Codex's default sandbox (`--full-auto`) uses Linux namespaces that:
1. **Block GPU access** — `cudaGetDeviceCount()` fails with error 304 because `/dev/nvidia*` devices aren't visible inside the namespace
2. **Block writes to `~/.cache/uv`** — the `uv` package manager can't initialize its cache, so `uv run train.py` fails immediately

Since this runs on a dedicated VM with nothing else on it, bypassing the sandbox is safe. On a shared machine, you'd need to configure the sandbox to allow device access and cache writes.

### Why one experiment per codex call?

We tried giving the agent a "loop forever" prompt, but:
- Context windows fill up after a few iterations of training output
- The agent tends to stop and ask if it should continue
- Error recovery is cleaner when each iteration is independent

One-experiment-per-call means each iteration starts fresh with a clean context, reads state from disk (git + results.tsv), and can't accumulate confusion from prior iterations.

## Pitfalls We Hit

These are the problems we encountered setting this up, so you don't have to:

### 1. Codex CLI doesn't read `OPENAI_API_KEY` from env

Even with the env var exported, `codex exec` returns 401. You must run:
```bash
echo "$OPENAI_API_KEY" | codex login --with-api-key
```
This stores the key in Codex's internal credential store.

### 2. CUDA blocked by sandbox

Codex's `--full-auto` mode (which uses `--sandbox workspace-write`) creates a Linux namespace that blocks access to GPU devices. Training crashes instantly with:
```
RuntimeError: Unexpected error from cudaGetDeviceCount().
Error 304: OS call failed or operation not supported on this OS
```
Fix: Use `--dangerously-bypass-approvals-and-sandbox` on dedicated GPU VMs.

### 3. uv cache permission denied

Even with `--full-auto --add-dir ~/.cache`, the sandbox blocks writes to nested paths:
```
error: Failed to initialize cache at `/home/user/.cache/uv`
  Caused by: failed to open file `/home/user/.cache/uv/sdists-v9/.git`: Permission denied
```
This is another reason to bypass the sandbox entirely on dedicated VMs.

### 4. Both models can't share one GPU simultaneously

Each training run uses 100% GPU for 5 minutes. You can't run two models in parallel on one GPU. Solutions:
- **Sequential** (what we did): Run model A for N hours, then model B for N hours
- **Two VMs**: True parallel, but costs 2×
- **Interleaved**: Alternate iterations — same conditions but slower

### 5. Non-interactive shell doesn't source `.bashrc`

Bash tool calls and tmux don't always source `.bashrc` (it has a non-interactive guard). The harness scripts check for `OPENAI_API_KEY` explicitly and fail fast if it's missing.

## Results

From our run comparing GPT-5.4 vs GPT-5.3-Codex (Spark/Cerebras) on a single H100. Both models started from the same commit (`c2450ad`) with identical, unmodified `train.py`.

### GPT-5.4 (6 hours, 54 iterations, 53 experiments)

```
commit   val_bpb   memory_gb  status   description
c2450ad  1.033848  44.0       keep     baseline
5a05b04  1.038355  44.0       discard  add 5% LR warmup
60c17f8  1.032859  44.0       keep     keep 10% LR floor during warmdown
0f7a84e  1.032669  44.0       keep     start warmdown later
f5100d3  1.032250  44.0       keep     delay warmdown slightly again
3643f61  1.031680  44.0       keep     raise final LR floor to 15%
6683761  1.029337  44.0       keep     shorten warmdown slightly
a8e6dda  1.022395  39.2       keep     reduce depth to 7
0e08a96  1.013308  39.1       keep     halve total batch size to double update frequency
ed450b6  1.011270  19.8       keep     halve total batch with 0.7x LRs and smaller device batch
7195765  1.010736  19.8       keep     lower LR scale for smaller batch
1f03c5f  1.010230  19.8       keep     lower LR scale another 10%
1dc230b  1.010159  19.8       keep     lower LR scale another 10% again
334d84a  1.009999  19.8       keep     lower LR scale another 10% once more
9d38d90  1.009684  19.8       keep     lengthen warmdown to 35%
a2feebf  1.009426  19.8       keep     lengthen warmdown to 37.5%
abcc35d  1.009307  19.8       keep     lengthen warmdown to 40%
eb3b7b6  1.009076  19.8       keep     lengthen warmdown to 42.5%
bf23fda  1.008841  19.8       keep     lengthen warmdown to 45%
a258850  1.008487  19.8       keep     lower Muon weight decay to 0.15 with small batch
f333383  1.008364  19.8       keep     lower Muon weight decay to 0.10
```

**Best val_bpb: 1.008364** (baseline: 1.033848, **-2.46%**)
Accept rate: 40% (21/53). 1 crash, 31 discarded.

GPT-5.4 made a bold architectural move mid-run: it reduced model depth from 8 to 7 layers and halved the batch size, trading model size for more gradient updates per 5-minute window. That single insight (1.033 → 1.013) was worth more than all of Spark's 57 experiments combined. It then systematically hill-climbed LR scaling and warmdown scheduling to push further to 1.008.

### GPT-5.3-Codex / Spark (6 hours, 57 iterations, 57 experiments)

```
commit   val_bpb   memory_gb  status   description
c2450ad  1.040068  44.0       keep     baseline
b0664e5  1.038591  44.0       keep     shorten LR warmdown ratio 0.5->0.2
7f7facb  1.038025  44.0       keep     extend LR warmdown ratio 0.2->0.25
2f5bad1  1.037462  44.0       keep     extend LR warmdown ratio 0.25->0.30
c335b68  1.031333  44.0       keep     tune warmdown ratio 0.30->0.29
ca9f304  1.030539  44.0       keep     ramp Muon momentum to 0.95 over 200 steps
7227edc  1.030434  44.0       keep     tune Muon momentum ramp target 0.95->0.94
bb218a4  1.030148  44.0       keep     lower Muon momentum ramp start 0.85->0.83
```

**Best val_bpb: 1.030148** (baseline: 1.040068, **-0.95%**)
Accept rate: 16% (9/57). 48 discarded.

Spark only tuned hyperparameters — LR warmdown ratios, Muon momentum ramp, weight decay. It never attempted architectural changes like reducing depth or changing batch size. It found a novel Muon momentum ramp trick that 5.4 didn't explore, but stayed within a narrow optimization space.

### Head-to-head

| | **GPT-5.4** | **GPT-5.3-Codex (Spark)** |
|---|---|---|
| Total time | 6h | 6h |
| Iterations | 54 | 57 |
| Experiments | 53 | 57 |
| Accepted | 21 (40%) | 9 (16%) |
| Baseline val_bpb | 1.033848 | 1.040068 |
| **Best val_bpb** | **1.008364** | **1.030148** |
| Improvement | **-2.46%** | -0.95% |
| Avg time/iter | ~402s | ~383s |

### Key Observations

- **Reasoning quality matters more than speed.** GPT-5.4 landed 40% of its experiments vs 16% for Spark. More importantly, 5.4 made a creative architectural leap (reducing depth + halving batch size) that Spark never attempted. Spark stuck to safe hyperparameter tweaks.
- **Spark was faster but not by enough.** ~19s saved per iteration (~383s vs ~402s) is about 5%. The fixed 5-minute training run dominates each iteration, so faster inference barely moves the needle.
- **5.4 found a deeper insight.** The key move was realizing that with a fixed 5-minute time budget, more gradient updates (smaller batch) beat a larger model. This required understanding the relationship between batch size, learning rate, and training dynamics — not just hill-climbing a single hyperparameter.
- **Both models found warmdown scheduling improvements**, but Spark got stuck there. 5.4 used it as a starting point, then moved on to structural changes.

## Customization

### Using different models

Any model available in your OpenAI account works:
```bash
./run_experiment.sh gpt-5.4 test1 4
./run_experiment.sh o3 test2 4
./run_experiment.sh gpt-5.4-pro test3 4
```

Check available models:
```bash
curl -s https://api.openai.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  | python3 -c "import sys,json; [print(m['id']) for m in json.load(sys.stdin)['data']]"
```

### Changing the time budget

```bash
./run_experiment.sh gpt-5.4 quick 2     # 2-hour run
./run_experiment.sh gpt-5.4 overnight 12 # 12-hour run
./launch_ab.sh 8                          # 8h per model, 16h total
```

### Modifying the agent prompt

Edit the `PROMPT` variable in `run_experiment.sh`. The current prompt tells the agent to:
- Run one experiment per call
- Read program.md for full protocol
- Log everything to results.tsv
- Keep/discard based on val_bpb

You can add domain-specific guidance (e.g., "focus on architecture changes, not just hyperparameters").

## Credits

- [Andrej Karpathy](https://github.com/karpathy) — autoresearch concept and repo
- [OpenAI Codex CLI](https://github.com/openai/codex) — the agent runtime
- The bash `while true` loop — the oldest trick in the book
