# Models

## The core idea

A standard transformer stores **2L separate coupling matrices** (W_Q, W_K at each layer). We discovered that these matrices share the same eigenvalue structure across layers (spectral correlation 0.98), even though the raw matrices look different. The coupling *rule* is invariant — only the residual stream changes.

The ORN **memoises** this rule as a single shared M = AB^T, applied at every layer with the same parameters but different input. This saves O(Ld²) parameters which are redistributed to model width.

## Model comparison

| Model | Coupling | Key feature | Use case |
|-------|----------|-------------|----------|
| **ORN** | SharedM (AB^T) | Proven at 91M scale | Main architecture |
| **ORNV2** | SharedM + GQA + RoPE | Modern defaults | V2 training |
| **CORN** | SharedM with t-scaling | Semigroup dynamics | Diffusion-style generation |
| **Transformer** | Per-layer W_Q, W_K | Standard | Fair comparison baseline |

## ORN (V1)

```python
from orn import ORN, ORNConfig

config = ORNConfig(d_model=512, n_layers=24, n_heads=8, d_corr=64,
                   vocab_size=50257, seq_len=256, version=1)
model = ORN(config)
```

Uses LayerNorm, GELU FFN, learned positional embeddings. This is the proven architecture that achieved val 3.35 on OpenWebText at 91M params.

## ORNV2

```python
from orn import ORNV2, ORNConfig

config = ORNConfig(d_model=576, n_layers=24, n_q_heads=9, n_kv_heads=3,
                   d_head=64, d_corr=64, vocab_size=49152, seq_len=2048, version=2)
model = ORNV2(config)
```

Modern defaults: RMSNorm, SwiGLU, RoPE, GQA. Designed for V2-125M training.

## CORN

```python
from orn import CORN, CORNConfig

config = CORNConfig(d_model=192, n_layers=8, n_heads=4, d_corr=48,
                    vocab_size=50257, seq_len=128, sg_lambda=0.05)
model = CORN(config)

# Standard forward
logits, loss = model(x, targets=y, t=1.0)

# Semigroup loss
sg_loss = model.semigroup_loss(x)

# Diffusion-style generation
output = model.generate_diffusion(seq_len=64, t_schedule=[2.0, 1.5, 1.0, 0.5])
```

CORN adds t-parameterised coupling: Q = h @ (t·A), K = h @ (t·B). The semigroup loss enforces M(t1+t2) ≈ M(t1) ∘ M(t2). At inference, iterating from high t to low t performs coarse-to-fine refinement using M's eigenspectrum as a natural noise schedule.

## Why asymmetric M?

Early experiments showed symmetric M = LL^T fails on autoregressive tasks. The asymmetry ||M - M^T|| / ||M|| is typically ~1.4, meaning M is far from symmetric. This makes sense: autoregressive coupling is inherently directional (past → future, not symmetric).

## Spectral diagnostics

All ORN/CORN models have `spectral_diagnostics()` and `count_params()`:

```python
spec = model.spectral_diagnostics()
# {eff_rank_90: 19, condition_number: 1450, asymmetry: 1.4, top_svs: [...]}

params = model.count_params()
# {total: 91000000, coupling: 524288, correction: 1200000, coupling_pct: 0.6}
```

The key diagnostic is **effective rank at 90% energy** — this is how many coupling dimensions the model actually uses. V1 converges to rank 19-20 out of 512.
