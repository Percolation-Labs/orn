# The coupling rule is the same in every layer — why transformers waste most of their attention parameters

A standard transformer stores two coupling matrices per layer: `W_Q` and
`W_K`. Multiplied together — `M_l = W_Q^T W_K` — they define the coupling
geometry that turns one token's representation into an attention weight
for another. With `L` layers and model dimension `d`, that's `O(L d²)`
parameters dedicated to coupling.

Here's the thing: you don't need most of them.

## The motivating observation

Take any pretrained transformer. Extract `M_l` at every layer. Look at
the sorted eigenvalue magnitudes. They all look like this:

![per-layer eigenspectra](figs/COU-02_coupling_manifold_8d.png)

The left panel overlays the curves — you can barely tell the layers apart.
The right panel is the Spearman correlation of those curves between each
pair of layers. Mean off-diagonal: **0.99**.

The raw matrices are different (cosine similarity on the order of 0.01).
But the *spectral structure* — the coupling rule — is invariant. Each
layer is spending `d²` parameters to re-encode the same rule.

## What to do about it

Replace the per-layer `W_Q / W_K` with a single shared pair `A, B ∈ R^{d×d}`,
applied at every layer to the residual stream:

```
Q_l = h_l · A       K_l = h_l · B       M = AB^T
```

The rule is memoised. The *state* (residual stream `h_l`) is what changes
with depth, and that's a free variable the transformer already updates.
The parameter budget drops from `O(Ld²)` to `O(d²)` for coupling. In a
24-layer model at d=576, that's a 24× reduction on 660k params per
layer — roughly 15M parameters freed up for width.

## Does it work?

Two predictions fall out:

1. **M should develop structure during training.** If coupling is
   genuinely low-rank, a trained `M` should have an effective rank well
   below `d`. In a 91M-parameter ORN at `d=512`, the effective rank
   crystallises to about 20 within the first 200M tokens of training.
   Roughly 4% of the full dimensionality.

2. **A frozen `M` should transfer.** If the coupling rule is the invariant
   part, you should be able to train a fresh model on a new dataset
   with `M` locked at a value taken from a different training run.
   That model should match a fully-trained one within a couple of percent
   of validation loss. Measured: 3.40 vs 3.35 on OpenWebText, i.e. within
   2%.

Both hold.

## Reproducing the figure

The figure above is produced by the orn reproduce suite in `Percolation-Labs/orn`:

```
pip install -e .
orn reproduce COU-02 --save-plots figs/
```

About three seconds on a laptop. The full tier of "fast" reproductions
(spectral universality across seeds, single-M recovery of per-layer
coupling, the colour-matching compression result, etc.) all run under
five minutes.

A heavier version on an actual pretrained GPT-2 — `SPE-03` in the
registry — needs the `.[data]` extras but produces the same figure on
the weights of a real 124M model.

## What this isn't

It isn't an argument that per-layer variation is useless. The *response*
heads (`W_V`, `W_O`, the FFN) still change layer-to-layer — different
layers apply the coupling rule to different transforms of the residual.
What the shared-M finding says is narrower: the rule itself is one
object, not L copies of it.

---

*Source code and published checkpoints: [github.com/Percolation-Labs/orn](https://github.com/Percolation-Labs/orn).*
*Try it on the 605M checkpoint: `orn predict --checkpoint orn-v3-605m`.*
