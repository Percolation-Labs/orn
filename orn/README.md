# orn — core package

Importable from `orn.*`. Everything the CLI and `tests/` need lives here.

```
orn/src/orn/
├── cli.py                CLI entry point (`orn …`)
├── models/
│   ├── layers.py         RMSNorm, SwiGLU, RoPE, PerturbativeCorrection
│   ├── orn.py            ORN V1 (LayerNorm/GELU) + V2 (RMSNorm/SwiGLU/RoPE/GQA)
│   ├── orn_v3.py         V3 (FullORN): V2 + single wide shared FFN
│   ├── corn.py           Composable ORN with semigroup dynamics
│   ├── lorn.py           VQREncoder + HypernetBridge + ORNV3 (LORN v4)
│   └── transformer.py    Vanilla baseline
├── training/
│   ├── trainer.py        Training engine (cosine LR, grad accum, amp, …)
│   ├── losses.py         lm_ce, lm_ce+lorn_vq, lm_ce+corn_ck dispatchers
│   ├── config_loader.py  JSON preset loader
│   └── configs/*.json    Named training presets
├── data/                 Prepare + ShardedDataLoader + synthetic tasks
├── diagnostics/          Spectral, coupling, plots
├── eval/                 Perplexity + lm-eval harness wrapper
├── reproduce/
│   ├── registry.py       REGISTRY of named key results
│   └── results/          One module per result (SPE-01, COU-*, …)
└── utils/                env (.env loader), device, io, hf (tokenizer/push/pull)
```

## Model dispatch

Every model is built via `build_model(arch, model_kw)` — this is what the
trainer uses so the JSON configs stay uniform.

```python
from orn import build_model
m = build_model("orn_v3", {"d_model": 576, "n_layers": 24,
                            "n_q_heads": 9, "n_kv_heads": 3, "d_head": 64,
                            "vocab_size": 49152, "seq_len": 2048,
                            "ffn_width_mult": 6, "d_corr": 64})
```

## Loss dispatch

```python
from orn.training import get_loss_fn
fn = get_loss_fn("lm_ce+lorn_vq")         # default: lm_ce
loss, logs = fn(model, x, y)
```

## Reproduce a key result

```python
from orn.reproduce import get
r = get("SPE-01")                          # or "spectral_universality"
print(r.run())
```
