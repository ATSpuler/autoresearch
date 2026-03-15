# Autoresearch Session — mar15

Read program.md, train.py, and prepare.py in full before doing anything.

## State

Branch: autoresearch/mar15 (forked from mar12 best)
Baseline (original): 1.3145
Current best: 1.1890 (after 42 experiments on mar12)
Steps per run: ~233 | VRAM: ~6.0GB | Noise floor: ±0.001

## Locked-in values (do NOT revert)

- TOTAL_BATCH_SIZE = 2**18
- MLP_RATIO = 3
- DEVICE_BATCH_SIZE = 16

## Setup

1. `git checkout -b autoresearch/mar15` (if not already done)
2. Verify `~/.cache/autoresearch/` has data + tokenizer
3. Create results.tsv with header row
4. First run: baseline on current code to anchor this session's numbers

## Priority experiments (in order)

### P1: TOTAL_BATCH_SIZE 2^17 (RETEST — highest leverage)
Previously failed at exp24 (1.2109 vs 1.2050) BEFORE MLP_RATIO=3 was applied.
With MLP_RATIO=3 the model is lighter → each step is faster → 2^17 gives ~466 steps.
Double the optimizer steps is the single biggest potential gain.
If it works, this becomes the new locked-in value.

### P2: Muon momentum warmup horizon 300→150 (BUG FIX)
Current code:
```python
def get_muon_momentum(step):
    frac = min(step / 300, 1)
    return (1 - frac) * 0.85 + frac * 0.95
```
With 233 steps, momentum never reaches 0.95 — it peaks at 0.934.
The model trains its ENTIRE life in warmup. This was never intentionally tested.
Try `step / 150` so warmup completes at step 150, leaving ~80 steps at full momentum.
This is nearly free — just a constant change. Test independently.

### P3: VE gate scale 2→3 (RETEST — match nanochat upstream)
Nanochat master uses `gate = 3 * sigmoid(...)`, your code uses 2.
Was borderline (Δ=0.0028) at 108 steps. Now at 233+ steps, may cross threshold.
One-line change in CausalSelfAttention.forward.

### P4: MLP c_fc init 0.5x (RETEST — match nanochat upstream)
Nanochat master:  `uniform_(-s * 0.5, s * 0.5)` for c_fc
Your code:        `uniform_(-s, s)`
Was "worse" at 108 steps with MLP_RATIO=4. Context has changed (MLP_RATIO=3, 233 steps).
The smaller init stabilizes early training when MLP is narrower.

### P5: depth 7 (fewer params → faster steps → more steps)
Test if removing one layer gives a net step gain that outweighs capacity loss.

### P6: Gradient clipping on AdamW groups
No gradient clipping exists anywhere. Muon self-normalizes via orthogonalization,
but AdamW groups (embedding LR=0.6!) have no clip. Try clip_grad_norm_=1.0 on
AdamW params only. Cheap insurance.

### P7: Combinations
If P1 works (2^17), immediately retest P2 on top of it.
If P1+P2 both work, try P3 (VE gate 3) on the combined code.
Stack winners aggressively.

## Context the agent should know

- This model is STEP-STARVED: 50M params, only 233 steps. Every change that
  increases step count wins. Every change that increases compute-per-step loses.
- The embedding norm `x = norm(x)` right after wte means embedding magnitude
  is irrelevant — only direction matters. Don't waste experiments on init std.
- The `adam_betas` parameter passed to setup_optimizer is dead code — each group
  has hardcoded betas already. Ignore it.
- Cosine warmdown was tested and is worse than linear. Don't retry.
- Matrix LR 0.10 is the sweet spot. 0.12 was worse. Don't sweep LR.

## After P1-P7: free exploration ideas

- TOTAL_BATCH_SIZE 2^16 if 2^17 works (push step count even further)
- MLP_RATIO 2 (extreme step gain, may lose too much capacity)
- n_kv_head < n_head (GQA — less attention compute per step)
- HEAD_DIM 64 (more heads, different attention geometry)
- WARMDOWN_RATIO sweep around 0.20-0.30
- RoPE theta 50000 (modest increase, not the 100K that failed before)
- Embedding LR 0.6→0.8 or 0.4 sweep

## Rules

- Noise floor ±0.001. Only keep if Δ > 0.003.
- Log every experiment to results.tsv (tab-separated, 5 columns).
- git commit before each run. Keep on improvement, reset on regression.
- Loop forever. Do not ask permission to continue.
- If a run exceeds 20 minutes wall clock, kill it.
- Each run: `uv run train.py > run.log 2>&1`
- Check: `grep "^val_bpb:\|^peak_vram_mb:" run.log`
