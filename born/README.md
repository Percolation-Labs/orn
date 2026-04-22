# born — basic ORN paper

Thin folder for the baseline ORN paper (lab notebook + scaling results).
Depends on `orn` for everything; only the paper source and paper-specific
experiments live here.

## Results reproduced here (all via `orn reproduce <code>`)

| Code   | What                                                    |
|--------|---------------------------------------------------------|
| SPE-01 | Spectral universality (5 pretrained families)           |
| SPE-02 | M crystallises to rank ~20 at 91M (slow tier)           |
| COU-01 | One shared M recovers most of GPT-2 attention           |
| COU-02 | Coupling manifold is low-dimensional                    |
| COU-03 | AB^T works; symmetric LL^T fails                        |
| COM-01 | 16× coupling compression, no quality loss               |

## Training commands that produced the reference numbers

```bash
orn train --config orn_v2_108m_8b     # 42h on 4090; val 3.007, HellaSwag 33.4
orn train --config orn_v3_61m_50b     # 6x wide shared FFN; half GPT-2 params
```

## Paper source

`paper/orn_lab_notebook.tex` (drop in when porting from stig-res).

## Custom experiments

Add only what is paper-specific. Everything general goes in `orn/reproduce/`.
