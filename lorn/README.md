# lorn — lateralised two-hemisphere paper

Thin folder for the LORN paper (concept-circuit battery + gauge analysis).
Depends on `orn` for everything; only paper source and paper-specific
experiments live here.

## Architecture (recap)

R (observer) → Bridge (hypernet) → L (predictor)

- R is a `VQREncoder`: attention + VQ codebook + entropy anti-collapse.
- Bridge produces per-sample low-rank deltas for L's attention output.
- L is an ORNV3 (shared M + wide shared FFN + per-layer corrections).

## Results reproduced here

| Code   | What                                             |
|--------|--------------------------------------------------|
| LOR-01 | VQ codebook fills, entropy stays up, no collapse |
| LOR-02 | Hypernet bridge beats AdaLN (slow tier, CC-18)   |

## Training

```bash
orn train --config lorn_v4_66m_100m     # 4h on 4090; first Mode A test
```

Monitor: `codes_used / codebook_size` climbs to ~K; `max_cluster_frac`
drops below 2/K; `val_ce` drops below a matched `orn_v3_61m_2b` run.

## Paper source

- `paper/two_hemisphere_lorn_gauge.tex` — current canonical LORN paper (v3/gauge).
- `paper/concept_circuits.tex` — the concept-circuit battery (CC-1..CC-23).
- `paper/two_hemispheres.tex` — legacy comprehensive draft, preserved for reference.
- Rendered PDFs alongside each.

## Key do/don't

- **Do** train R+L end-to-end from scratch.
- **Don't** mix VQ with contrastive on a shared encoder (CC-15 kill).
- Codebook size K ≤ expected latent class count (over-sizing hurts purity, CC-16).
