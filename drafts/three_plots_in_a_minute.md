# Three small transformer-coupling plots you can reproduce in a minute

I have been building out a small research repo called ORN and one thing I wanted early was for every headline claim to be a one-line command away. Not a pointer to a notebook, not a frozen PDF, a command. This is a quick tour of three plots that fall out of the reproduce suite, why each one exists, and what they say.

All three use only torch, numpy, scipy, and matplotlib. All three finish in under ten seconds on a laptop CPU. Clone the repo, pip install, run.

```
git clone https://github.com/Percolation-Labs/orn.git
cd orn
pip install -e .
orn reproduce COU-02 COM-01 SPE-01 --save-plots figs/
```

The three plots below are exactly what that command drops into the figs directory.

## The motivating figure for shared M

The reason the repo exists. Train a small transformer. At each layer, extract M_l = W_Q^T W_K (the coupling matrix). Sort the eigenvalue magnitudes. Overlay them. Compute the Spearman correlation between each pair.

![Per-layer eigenvalue overlay plus cross-layer correlation heatmap](https://raw.githubusercontent.com/Percolation-Labs/orn/article-drafts/drafts/figs/COU-02_coupling_manifold_8d.png)

The left panel is the overlay. The curves sit on top of each other. The right panel is the pairwise Spearman. Mean off-diagonal is 0.999.

The raw matrices are nowhere near each other (cosine similarity on the order of 0.01), but the spectral structure is shared. The transformer is spending d^2 parameters per layer to re-encode the same rule. That is the observation that motivates memoising the coupling into a single shared M = A B^T.

> If the rule is invariant, you only need to parameterise it once.

## The compression claim, on a task where coupling is provably invariant

The risk with the figure above is that you might be seeing a training artifact, or a quirk of the toy model. So I built a control task where the coupling is invariant by construction: a colour matching task where the sequence structure is periodic and the same at every depth.

Then I trained two models. MEMOISE shares one A, B pair across all four layers. STORE gets a fresh pair at every layer. Everything else is held equal.

![MEMOISE vs STORE: training loss curves and coupling parameter counts](https://raw.githubusercontent.com/Percolation-Labs/orn/article-drafts/drafts/figs/COM-01_colour_matching_compression.png)

Left panel is the loss curve. The two models track each other, with STORE sometimes slightly faster in the middle of training. Both converge to roughly the same final loss.

Right panel is the coupling parameter budget. MEMOISE: 8,192. STORE: 32,768. A factor of four compression, and the only thing you lost is a handful of tenths in validation loss.

On the provably-invariant task, sharing M is free.

## Does this generalise across models

It is one thing to see the invariance on a single trained transformer. What happens when you train three different transformers with different seeds on the same data? If the coupling rule is actually invariant, the layer-averaged M should have similar spectra across models, not just across layers within one model.

![Sorted eigenvalue magnitudes of layer-averaged M, across three seeded models](https://raw.githubusercontent.com/Percolation-Labs/orn/article-drafts/drafts/figs/SPE-01_spectral_universality.png)

The three curves overlap. Mean pairwise Spearman is 0.9994.

At toy scale with random training data this is a weak check. The stronger version, using actual pretrained transformers (GPT-2, SmolLM2, Qwen2.5), lives as `SPE-03` in the repo and requires pip install -e '.[data]' for the HuggingFace fetchers. That slow tier result reports 0.93+ correlations on real pretrained weights, which is the number worth quoting.

## What the tooling looks like

Each result is a Python module under `orn/reproduce/results/` that exposes two functions: `run(device=None) -> dict` and optionally `plot(result, save_path=None) -> figure`. A registry maps short codes (SPE-01, COU-02, COM-01) to modules and marks which ones produce plots. The CLI flag `--save-plots DIR` writes one PNG per result.

You can also load a real published checkpoint and run it against the same diagnose tooling that produced the plots:

```
orn pull-checkpoint orn-v3-605m
orn diagnose --checkpoint orn-v3-605m
```

The largest published checkpoint is a 605M ORN V3 trained on 2B tokens of FineWeb-Edu. It is public on HuggingFace and the utility handles the download and the legacy state-dict remapping so you just get a loaded model back.

## Why this matters for a research repo

Reproducibility of the headline numbers is easy to write down and hard to keep alive. Code rot gets every project sooner or later. The pattern I like is making each claim its own test: a registry entry, a run function, a plot function, a pytest case. If a refactor breaks a claim, pytest tells me, and the plot in the readme stops matching what the code produces.

Whole repo and the checkpoints are at [github.com/Percolation-Labs/orn](https://github.com/Percolation-Labs/orn). Feedback welcome.
