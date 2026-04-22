# ORN — Orbital Response Network

A transformer variant where the per-layer query-key coupling `W_Q^T W_K` is
replaced by a **single shared** metric tensor `M = AB^T`, applied at every
layer to an evolving residual stream. The coupling *rule* is memoised; the
saved O(Ld²) budget is redistributed to model width. Empirically:
`M` crystallises to rank ~20/512 at 91M; frozen-M transfer matches full
training within 2%; spectral universality across GPT-2 / SmolLM2 / Qwen2.5.

## Layout

```
orn/             ← core library (models, trainer, CLI, reproduce suite)
born/            ← basic-ORN paper (thin; imports from orn)
lorn/            ← lateralised two-hemisphere paper (thin; imports from orn)
papers/drafts/   ← unlanded drafts
tests/           ← pytest (fast reproductions run by default)
```

## Install

```bash
pip install -e .                 # core
pip install -e '.[data]'         # + HuggingFace datasets / hub
pip install -e '.[eval]'         # + lm-eval-harness
pip install -e '.[vast]'         # + wandb, datasets (for cloud runs)
```

Requires Python ≥3.9 and PyTorch ≥2.1.

## Environment

Copy `.env.example` → `.env` and fill in:
- `HF_TOKEN`   HuggingFace token (pull SmolLM2 tokenizer, pull/push baselines)
- `HF_HOME`    model cache location
- `WANDB_API_KEY`  optional

## CLI

```bash
orn configs                         # list training presets
orn prepare tinystories             # download + tokenise
orn train --config dry_run          # ~30s pipeline test
orn train --config orn_v2_108m_8b   # production run
orn predict --checkpoint output/checkpoints/latest.pt --prompt "The"
orn diagnose --checkpoint output/checkpoints/latest.pt
orn eval --checkpoint output/checkpoints/latest.pt --tasks hellaswag,piqa,arc_easy
orn reproduce --list                # fast + slow reproducible results
orn reproduce SPE-01 COU-01 LOR-01
orn push --checkpoint output/checkpoints/latest.pt --repo you/orn-ckpt
orn pull --model smollm2-135m
```

## Reproducible key results

`orn/reproduce/` is the single source of truth. Each result is identified by a
code (SPE-01, COU-01, …) and a slug (spectral_universality, …). The fast tier
runs in under 5 minutes on a laptop and is the default `pytest` target;
the slow tier is opt-in via `pytest -m slow_reproduce` or `orn reproduce --slow`.

```bash
pytest tests/                     # all smoke + fast-reproduce
pytest tests/ -m reproduce        # fast reproductions only
pytest tests/ -m slow_reproduce   # overnight runs
```

| Code   | What it shows                                       | Runtime |
|--------|-----------------------------------------------------|---------|
| SPE-01 | Eigenspectra correlate across pretrained families  | 60s     |
| COU-01 | One shared M recovers most of GPT-2 attention       | 15s     |
| COU-02 | Coupling manifold is low-dimensional                | 30s     |
| COU-03 | AB^T works; symmetric LL^T fails                    | 90s     |
| COM-01 | 16× coupling compression, no quality loss           | 120s    |
| OPE-01 | Function-token attention stable; content variable   | 60s     |
| OPE-04 | ORN = Lie-Trotter splitting for exp(t(A+B))         | 30s     |
| LOR-01 | VQ codebook fills, no collapse                      | 60s     |

Plus slow-tier SPE-02 (M-crystallisation at scale), COM-01/scaling, etc.

## Models

| Arch            | Key feature                                         |
|-----------------|-----------------------------------------------------|
| `orn_v1`        | LayerNorm, GELU, learned-pos. Proven at 91M.        |
| `orn_v2`        | RMSNorm, SwiGLU, RoPE, GQA. Modern defaults.        |
| `orn_v3`        | V2 + single wide shared FFN.                        |
| `corn`          | Semigroup-dynamics variant (time-scaled coupling).  |
| `lorn_v4`       | Two-hemisphere: VQ observer + hypernet + ORNV3.     |
| `transformer`   | Vanilla baseline for controlled comparison.         |

## Papers

- `born/` — basic ORN / lab-notebook paper; covers SPE-01/02, COU-\*, COM-01.
- `lorn/` — two-hemisphere paper; covers LOR-\* and the concept-circuit battery.
- `papers/drafts/` — semigroup_deep_dive, multimodal_orbits, m_calculus, tutorial, etc.

## License

MIT.
