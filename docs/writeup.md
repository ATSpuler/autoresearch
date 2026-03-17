# 156 Experiments on RTX 3060: Autoresearch Results

**repo**: [github.com/ATSpuler/autoresearch](https://github.com/ATSpuler/autoresearch)  
**branches**: `autoresearch/mar12` (exp 1–42), `autoresearch/mar15` (exp 43–156)  
**hardware**: Single RTX 3060 12GB, Ubuntu 22.04  
**time budget**: 600s per run (upstream default is 300s, doubled for this hardware)  
**noise floor**: ±0.001 val_bpb between identical runs  

## Result

```
val_bpb:  1.3145 → 1.1460  (12.8% improvement)
params:   50M → 36M
steps:    109 → 372
VRAM:     6.0 GB → 4.1 GB
```

The final model is smaller, faster per step, uses less memory, and scores better.

## Winning trajectory

Every kept experiment, in order:

| # | val_bpb | Δ | Steps | Change |
|---|---|---|---|---|
| baseline | 1.3145 | — | 109 | upstream defaults |
| 1 | 1.3065 | +0.008 | 109 | per-group AdamW betas + weight decay |
| 2 | 1.3040 | +0.003 | 109 | Muon beta2 0.95→0.9 |
| 5 | 1.2878 | +0.016 | 109 | warmdown 0.35 + final LR frac 0.05 |
| 6 | 1.2704 | +0.017 | 109 | QK post-scale q,k *= 1.15 |
| 16–21 | 1.2481 | +0.022 | 109 | matrix LR sweep 0.04→0.10 + warmdown 0.25 |
| 23 | 1.2050 | +0.043 | 204 | TOTAL_BATCH_SIZE 2^19→2^18 |
| 36 | 1.2018 | +0.003 | 204 | x0_lambdas init 0.1→0.0 |
| 42 | 1.1890 | +0.013 | 233 | MLP_RATIO 4→3 |
| — | — | — | — | *session break, new branch* |
| 44 | 1.1851 | +0.004 | 233 | warmdown 0.25→0.50 |
| 46 | 1.1819 | +0.003 | 233 | warmdown 0.50 |
| 50 | 1.1712 | +0.011 | 261 | MLP_RATIO 3→2 |
| 66 | 1.1656 | +0.006 | 275 | short window 1024→512 |
| 78 | 1.1596 | +0.006 | — | combo: window 256 + warmdown 0.60 + all-S pattern |
| 80 | 1.1557 | +0.004 | 338 | MLP_RATIO 2→1 |
| 91 | 1.1542 | +0.002 | 372 | depth 8→7 |
| 98 | 1.1510 | +0.003 | 372 | VE on all 7 layers (was alternating) |
| 105 | 1.1495 | +0.002 | 372 | remove QK post-scale 1.15 (simplification) |
| 125 | 1.1460 | +0.004 | 372 | Muon momentum 0.90 + VE gate channels 16 |

Pattern: almost every major win either increased step count or simplified the model.

## What's essential (removing any of these causes large regression)

| Component | Regression when removed |
|---|---|
| Value embeddings | +0.024 |
| VE gating mechanism | +0.036 |
| QK-norm | +0.031 |
| x0_lambdas (highway connection) | +0.009 |
| Softcap | +0.008 |
| MLP (RATIO=0) | +0.047 |
| Forced long-window last layer | +0.023 |

## What never mattered (noise at every configuration tested)

VE gate scale (1, 3, 4 — 2 always optimal), RoPE theta changes, SiLU/SwiGLU (relu² always wins), parallel attn+MLP, GQA, HEAD_DIM 64, any form of LR warmup, gradient clipping on AdamW groups, VE gate init schemes, embedding init std.

## One surprising observation

QK post-scale (q,k *= 1.15 after normalization) was a clear win at 109 steps (+0.017). At 372 steps, removing it *improved* results (+0.002). Same intervention, opposite effect at different step counts. I don't fully understand why yet — noting it as an empirical finding.

## Final model vs upstream defaults

| | Upstream (H100) | Original (RTX 3060) | Final (RTX 3060) |
|---|---|---|---|
| val_bpb | ~0.997 | 1.3145 | 1.1460 |
| steps | 953 | 109 | 372 |
| params | 50M | 50M | 36M |
| MLP ratio | 4 | 4 | 1 |
| depth | 8 | 8 | 7 |
| attn window | 1024 | 1024 | 256 |
| VE layers | alternating | alternating | all |
| VRAM | — | 6.0 GB | 4.1 GB |

The optimal configuration on RTX 3060 looks nothing like the H100 defaults. Smaller, shallower, shorter attention, more value embedding layers.

## Reproduce

```bash
git clone https://github.com/ATSpuler/autoresearch
cd autoresearch
git checkout autoresearch/mar15
uv sync
uv run prepare.py
uv run train.py  # 10 min, expect val_bpb ≈ 1.146
```

Full results: `results.tsv` and `results_mar12.tsv` on the branch.  
Every experiment is a git commit with a one-line description.

## Methodology

The experiment loop followed [karpathy/autoresearch](https://github.com/karpathy/autoresearch)'s program.md with an LLM-generated research agenda (Claude Opus) curated and directed by a human between sessions. The agent (Claude Code) ran autonomously inside a container.

## Session breakdown

| Session | Duration | Experiments | Kept | Hit rate |
|---|---|---|---|---|
| 1 (mar12) | ~19h | 5 | 3 | 60% |
| 2 (mar12 cont.) | ~8.5h | 42 | 8 | 19% |
| 3 (mar15) | ~25h+ | 114 | 12 | 10.5% |
| **Total** | **~52h** | **~161** | **23** | **14%** |

If you see patterns in the results I'm missing, I'd love to hear them.
