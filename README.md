# ORN — Orbital Response Network

A transformer variant where the per-layer query-key coupling `W_Q^T W_K` is
replaced by a **single shared** metric tensor `M = AB^T`, applied at every
layer to an evolving residual stream. The coupling *rule* is memoised; the
saved O(Ld²) budget is redistributed to model width. Empirically:

- `M` crystallises to rank ~20/512 at 91M (most of the 512 dimensions are redundant).
- Frozen-M transfer matches full training within 2% on a new dataset.
- Spectral correlation ≥ 0.93 across GPT-2 / SmolLM2 / Qwen2.5 layers (the motivating invariance).

## 5-minute walkthrough

From a clean clone:

```bash
git clone https://github.com/Percolation-Labs/orn.git
cd orn
pip install -e .                                  # core (torch, numpy, scipy, matplotlib, tiktoken)

# 1) Reproduce the motivating figure for shared M  (~3s, no data needed)
orn reproduce COU-02 --save-plots figs/
#   → figs/COU-02_coupling_manifold_8d.png
#   Per-layer W_Q^T W_K eigenspectra overlap, cross-layer ρ ≈ 0.99.

# 2) Run the whole fast-tier suite                 (~15s, 8 results)
orn reproduce SPE-01 COU-01 COU-02 COU-03 COM-01 OPE-01 OPE-04 LOR-01 \
              --save-plots figs/

# 3) Full pipeline dry run                         (~2s, auto-synthesises data)
orn train --config dry_run --device cpu

# 4) Spectral diagnostics on the checkpoint
orn diagnose --checkpoint output/checkpoints/final.pt

# 5) The REAL motivating figure on pretrained GPT-2  (~2 min, first run downloads)
pip install -e '.[data]'
orn reproduce SPE-03 --slow --save-plots figs/
```

## Layout

```
orn/           core library  (models, trainer, CLI, reproduce suite)
born/          basic-ORN paper folder (thin; imports from orn)
lorn/          lateralised two-hemisphere paper folder (thin)
papers/        unlanded drafts
tests/         pytest; fast reproductions run by default
```

## Install options

```bash
pip install -e .                 # core
pip install -e '.[data]'         # + HuggingFace datasets/transformers (for SPE-03, prepare)
pip install -e '.[eval]'         # + lm-eval-harness (for `orn eval`)
pip install -e '.[vast]'         # + wandb (for cloud training)
```

Python ≥ 3.9, PyTorch ≥ 2.1.

## Environment

Copy `.env.example` → `.env` (git-ignored) and fill in:

```
HF_TOKEN=hf_...          # HuggingFace — pulls SmolLM2 tokenizer, pretrained baselines, pushes checkpoints
HF_HOME=~/.cache/huggingface
WANDB_API_KEY=...        # optional
```

Most fast reproduce results need no token. SPE-03 (pretrained GPT-2) does
need `HF_TOKEN` for gated models, but `gpt2` itself is public.

## CLI

```
orn configs                                   list training presets
orn prepare <dataset>                         download + tokenise (tinystories|openwebtext|fineweb-edu)
orn train --config <NAME|PATH> [overrides]    train a model
orn predict --checkpoint PATH --prompt …      generate text
orn diagnose --checkpoint PATH [--plot F]     spectral diagnostics on M
orn eval --checkpoint PATH --tasks …          hellaswag, piqa, arc_easy
orn reproduce [--list] [names…] [--save-plots DIR] [--slow]
orn push --checkpoint PATH --repo USER/NAME   upload to HuggingFace
orn pull --model NAME                         gpt2 | smollm2-135m | pythia-70m | pythia-410m
```

Training presets:

| Preset              | Arch     | Notes                                        |
|---------------------|----------|----------------------------------------------|
| `dry_run`           | orn_v1   | ~2s; auto-synthesises data if none present   |
| `orn_v2_108m_8b`    | orn_v2   | 4090/42h, val 3.007, HellaSwag 33.4          |
| `orn_v3_61m_50b`    | orn_v3   | 6× wide shared FFN; half GPT-2 params        |
| `lorn_v4_66m_100m`  | lorn_v4  | Two-hemisphere first run (~4h)               |

CLI overrides on any preset: `--seq-len`, `--batch-size`, `--grad-accum-steps`,
`--max-tokens`, `--d-model`, `--n-layers`, `--no-compile`, `--no-mixed-precision`, `--device`, …

## Reproducible key results

Each result lives in `orn/reproduce/results/<slug>.py` with a short
`run(device=None) -> dict` and (for most) a `plot(result, save_path=None) -> fig`.
`registry.py` maps codes to modules. `orn reproduce --list` shows which have plots.

### Fast tier (default pytest, ~15s total)

| Code   | Plot | What it shows                                      |
|--------|:----:|----------------------------------------------------|
| SPE-01 |  ✓   | Eigenspectra correlate across seeded models        |
| COU-01 |  ✓   | One shared M recovers most of per-layer W_Q^T W_K  |
| COU-02 |  ✓   | Coupling manifold is low-dimensional (**the motivating figure**) |
| COU-03 |      | AB^T works; symmetric LL^T fails                   |
| COM-01 |  ✓   | 16× coupling compression, MEMOISE ≈ STORE          |
| OPE-01 |      | Function-token attention stable; content variable  |
| OPE-04 |      | ORN = Lie-Trotter splitting of exp(t(A+B))          |
| LOR-01 |      | VQ codebook fills, no collapse                     |

### Slow tier (opt-in; `orn reproduce --slow`)

| Code   | Plot | What it shows                                      |
|--------|:----:|----------------------------------------------------|
| SPE-02 |  ✓   | M crystallisation trajectory (eff-rank drop over training) |
| SPE-03 |  ✓   | Real GPT-2 per-layer W_Q^T W_K invariance (needs `.[data]`) |

### Testing

```bash
pytest tests/                      # all smoke tests + fast reproductions (~11s)
pytest tests/ -m reproduce         # fast reproductions only
pytest tests/ -m slow_reproduce    # slow tier (skipped by default)
```

## Models

| Arch            | Key feature                                         |
|-----------------|-----------------------------------------------------|
| `orn_v1`        | LayerNorm, GELU, learned-pos; proven at 91M.        |
| `orn_v2`        | RMSNorm, SwiGLU, RoPE, GQA; modern defaults.        |
| `orn_v3`        | V2 + single wide shared FFN.                        |
| `corn`          | Semigroup-dynamics variant (time-scaled coupling).  |
| `lorn_v4`       | Two-hemisphere: VQ observer + hypernet + ORNV3.     |
| `transformer`   | Vanilla baseline for controlled comparison.         |

Every model is built through `build_model(arch, config)` so the JSON
config schema stays uniform.

## Papers

- `born/` — basic ORN / lab-notebook paper; covers SPE-\*, COU-\*, COM-01.
- `lorn/` — two-hemisphere paper; covers LOR-\* and the concept-circuit battery.
- `papers/drafts/` — semigroup_deep_dive, multimodal_orbits, m_calculus, tutorial, etc.

## License

MIT.
