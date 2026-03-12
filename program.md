# autoresearch

This is an experiment to have the LLM do its own research.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar12`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data prep, tokenizer, dataloader, evaluation. Do not modify.
   - `train.py` — the file you modify. Model architecture, optimizer, training loop.
4. **Verify data exists**: Check that `~/.cache/autoresearch/` contains data shards and a tokenizer. If not, tell the human to run `uv run prepare.py`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Hardware context

This machine runs a single **NVIDIA RTX 3060 12GB** GPU (PCIe Gen 2). Key constraints:
- VRAM budget: 12GB hard limit. Current baseline uses ~6.0GB. Do not exceed 11GB.
- DEVICE_BATCH_SIZE = 16 is the tuned value for this GPU. Do not increase above 24.
- TIME_BUDGET = 600s (10 minutes). This is already set in prepare.py.
- Expected throughput: ~86K tok/sec, ~109 steps per run, ~57M tokens per run.
- MFU is ~2% — this is normal for this GPU, not a bug.

## Experimentation

Each experiment runs on a single GPU. The training script runs for a **fixed time budget of 10 minutes** (wall clock training time, excluding startup/compilation). You launch it simply as: `uv run train.py`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, data loading, tokenizer, and training constants (time budget, sequence length, etc).
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the evaluation harness. The `evaluate_bpb` function in `prepare.py` is the ground truth metric.

**The goal is simple: get the lowest val_bpb.** Since the time budget is fixed, you don't need to worry about training time — it's always 10 minutes. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size. The only constraint is that the code runs without crashing and finishes within the time budget.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful val_bpb gains, but do not exceed 11GB peak.

**Noise floor**: On this hardware, identical runs vary by ~±0.001 val_bpb. Therefore:
- Improvement of > 0.003: real, keep it
- Improvement of 0.001-0.003: borderline, keep only if the change is also a simplification
- Improvement of < 0.001: noise, discard regardless
- Simplification with equal val_bpb: always keep

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Output format

Once the script finishes it prints a summary like this:

```
---
val_bpb:          1.311055
training_seconds: 601.2
total_seconds:    891.4
peak_vram_mb:     6150.2
mfu_percent:      2.09
total_tokens_M:   57.1
num_steps:        109
num_params_M:     50.3
depth:            8
```

This is what our baseline looks like on this RTX 3060 machine. The original repo's README shows H100 numbers (val_bpb ~0.997, 953 steps) — those are not our target. Our target is to beat **1.311055**.

You can extract the key metric from the log file:

```
grep "^val_bpb:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	val_bpb	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. val_bpb achieved (e.g. 1.234567) — use 0.000000 for crashes
3. peak memory in GB, round to .1f (e.g. 6.0 — divide peak_vram_mb by 1024) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	val_bpb	memory_gb	status	description
a1b2c3d	1.311055	6.0	keep	baseline
b2c3d4e	1.305200	6.0	keep	per-group AdamW betas
c3d4e5f	1.315000	6.0	discard	switch to GeLU activation
d4e5f6g	0.000000	0.0	crash	double model width (OOM)
```

## Research agenda

Work through this agenda in order. Each section represents a known area of improvement based on prior analysis. Validate each finding empirically — do not assume a change will help just because it is listed here. The experiment loop is the ground truth.

### Phase 1: Optimizer fixes (highest leverage — affects every gradient step)

The current train.py uses a single shared `adam_betas=(0.8, 0.95)` and `weight_decay=0.0` for ALL AdamW parameter groups. This is suboptimal. Each parameter group has different gradient statistics and needs different hyperparameters.

Hypotheses to test, roughly in priority order:

1. **Per-group AdamW betas**: Different parameter groups need different beta2 (memory horizon):
   - `lm_head`: try `(0.8, 0.96)` — dense updates, moderate memory
   - `embeddings`: try `(0.8, 0.995)` — sparse updates, need long memory for stability
   - `value_embeds`: try `(0.8, 0.995)` — same reasoning as embeddings
   - `resid_params` (scalars): try `(0.8, 0.95)` — current value, probably fine
   Test these together or individually.

2. **Weight decay on embeddings**: Currently `weight_decay=0.0` for ALL groups including `value_embeds` (16M parameters). This is likely causing overfitting. Try adding regularization:
   - `value_embeds`: weight_decay=0.01
   - `lm_head`: weight_decay=0.01
   - `resid_params`: weight_decay=0.05
   Keep embeddings (wte) at 0.0 or very small (0.001).

3. **Muon beta2**: Currently 0.95. Try 0.9 — faster second-moment decay = more aggressive matrix updates. This makes Muon adapt faster to gradient changes.

4. **Muon momentum warmup**: Currently warms to 0.95. Try warmer target: 0.97 over 400 steps.

5. **Weight decay schedule**: Currently constant. Try cosine decay to 0 over training. Intuition: early training benefits from regularization, late training (fine detail fitting) less so.

### Phase 2: Architecture fixes (structural issues)

6. **QK-norm post-scale**: After QK normalization, queries and keys have unit norm. This makes attention logits `q·k = cos(angle)` which is bounded in [-1, 1] — attention may be too diffuse (uniform). Try scaling: `q = q * 1.15; k = k * 1.15` after the norm. This sharpens attention without breaking the normalization's stability benefits.

7. **VE gate channels**: Currently `ve_gate_channels=32`. The gate uses the first 32 channels of x to compute per-head gating weights. Try reducing to 12 — fewer channels = less noise in gate computation, more stable value residual mixing.

8. **VE gate scale**: Currently `gate = 2 * sigmoid(...)` giving range (0, 2). Try `gate = 3 * sigmoid(...)` giving range (0, 3) — allows stronger mixing of value embeddings when the gate fires high.

9. **VE gate init**: Currently `zeros_init` so gates start at sigmoid(0)=0.5, scaled to 1.0 (neutral). Try small positive init `uniform_(0.0, 0.02)` so gates start slightly above neutral, biasing toward using value embeddings from the start.

10. **Embedding init std**: Currently `std=1.0`. Try `std=0.8` — smaller init reduces embedding magnitude, which can improve gradient flow through the embedding table in early training.

11. **MLP c_fc init**: Try initializing `c_fc` weights at 0.5× the normal scale. Smaller MLP input projection init can stabilize early training.

### Phase 3: Schedule and misc

12. **Warmup**: Currently ratio-based. Try absolute steps (e.g. 40 steps) — more predictable across different total step counts.

13. **Warmdown ratio**: Currently 0.5 (last 50% of training decays LR). Try 0.65 — longer decay = smoother final convergence.

14. **Final LR fraction**: Currently 0.0 (LR decays to zero). Try 0.05 — a small residual LR at the end can help.

15. **Logit softcap**: Already at 15 (post-fix value). Leave unless evidence suggests otherwise.

### Phase 4: Free exploration

After Phase 1-3, explore freely. Ideas to consider:
- Different activation functions (relu² is current — what about SwiGLU?)
- Attention pattern changes (current: SSSL — short/short/short/long)
- RoPE theta (currently 10000 — try 100000 for longer effective context)
- Depth/width tradeoff (current depth=8 — does depth=6 with wider width fit better in the token budget?)
- Learning rate magnitude sweeps
- Gradient clipping values
- Any ideas not listed above — be creative

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar12`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `uv run train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^val_bpb:\|^peak_vram_mb:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If val_bpb improved (lower), you "advance" the branch, keeping the git commit
9. If val_bpb is equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate.

**Timeout**: Each experiment should take ~10 minutes total (+ startup/compilation overhead of ~5 min). If a run exceeds 20 minutes wall clock, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

On this hardware (10 min per experiment) you can run approximately 4-5 experiments per hour, roughly 40 experiments overnight. Make each one count.
