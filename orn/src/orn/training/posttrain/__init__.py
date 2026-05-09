"""Post-training stages for the public ORN package.

Mirrors the pipeline in `drafts/post_training_a_600m_orn.md`:

| Module               | Stage                          | Article section                          |
|---------------------|---------------------------------|-------------------------------------------|
| `sft`               | SFT-1 instructions / SFT-2 chat | Stage 2 + 3                               |
| `dpo`               | Direct Preference Optimisation  | Stage 4 (helpful-assistant register)      |
| `long_context`      | PoSE + NTK-rescaled RoPE        | Stage 5 (positional reach)                |
| `fc_sft`            | Function-calling SFT (GlaiveAI) | Stage 6 (structured tool-call output)     |
| `data_fc`           | Glaive/xLAM/Gorilla data prep   | (data layer for the FC stage)             |

Each script is a standalone trainer that loads a checkpoint, runs the
stage-specific training loop, and saves the result. The CLI wires them as
`orn sft`, `orn dpo`, `orn long-ctx-ft`, `orn fc-sft`. Configs for each
stage live in `orn/training/configs/` (e.g. `sft_openhermes.json`).

The trainers were ported verbatim from the private research repo; the only
public-side changes are the model imports (`FullORN` → `ORNV3`) handled by
the shared `_compat` shim.
"""
