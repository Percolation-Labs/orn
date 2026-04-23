# The Orbital Response Network

I want to describe a small architectural change to transformer attention that I think is worth paying attention to. It is not a compression trick, even though it produces compression. The reason to care about it is what it tells us about what transformers already know.

## The observation

Every transformer layer learns its own pair of coupling matrices W_Q and W_K. The object that actually determines which tokens attend to which is not either matrix alone but the product M_l = W_Q^T W_K. If you look at the eigenvalue magnitudes of M_l at every layer of a trained transformer and plot them on top of each other, they line up.

![Per-layer coupling eigenspectra overlap; Spearman heatmap across layers](https://raw.githubusercontent.com/Percolation-Labs/orn/article-drafts/drafts/figs/COU-02_coupling_manifold_8d.png)

The raw matrices point in different directions. Cosine similarity between any two M_l matrices sits around 0.005. But the spectrum is the same everywhere. On a small trained transformer in the repo, the pairwise Spearman correlation of sorted eigenvalue magnitudes is 0.999 across layers. On real pretrained transformers (three GPT-2 sizes, SmolLM2-135M, Qwen2.5-0.5B), the correlation sits between 0.93 and 0.99.

M is a bilinear form. Its spectrum is invariant under orthogonal rotation. What the data is telling us is that every layer has learned the same bilinear form, up to rotation. The layers are not independent draws from a distribution of coupling rules. They are one rule, expressed in different coordinate frames, acting on a residual stream that rotates through those frames as depth increases.

I spent a while checking whether this is an artefact of training, a quirk of one model, or a feature of one scale. It reproduces across model families, across seeds, and across training runs. Same thing showing up every time.

![Layer-averaged M across three independent seeds; curves lie on top of each other](https://raw.githubusercontent.com/Percolation-Labs/orn/article-drafts/drafts/figs/SPE-01_spectral_universality.png)

## Exploiting it

If one rule is enough for every layer, parameterise it once. Replace per-layer W_Q and W_K with a single shared pair A, B in R^{d x d}, and define

```
Q_l = h_l · A       K_l = h_l · B       M = A B^T
```

applied at every layer of a stack. Per-layer variation comes from the residual stream h_l, which the transformer already updates layer by layer. Response heads (W_V, W_O, the FFN) stay per-layer, because the response to coupling legitimately varies with depth.

We call this an Orbital Response Network. One pair A, B governs coupling across the whole stack. The asymmetry matters: M = A B^T, not M = L L^T. Language coupling is directed (subjects and verbs couple differently depending on who is attending to whom), and symmetric coupling loses about 3.6% on validation on autoregressive tasks.

## How this is different from prior weight sharing

People have tied transformer weights before. ALBERT shares every parameter across layers and loses roughly 1.5% of quality in exchange for compression. MQA and GQA share keys and values across query heads, a different axis motivated by inference-time KV cache size, not by anything about the coupling rule. MASA decomposes attention matrices into shared dictionary atoms via learned dictionary learning. RRT (ICLR 2025) ties everything, then adds per-layer LoRA patches.

These are all "share a lot of parameters and hope quality survives." ORN does something different. It shares only the specific piece that the data says is invariant. Measurement first, architecture second. The spectral correlation across layers is north of 0.93 on pretrained models; that is the concrete observation. Response heads are still per-layer because depth-wise variation in the response is not invariant.

## Numbers

Two published checkpoints on HuggingFace, both public:

- `mr-saoirse/orn-v3-605m` is a 605M parameter ORN V3 (d=2048, L=32, GQA 32 over 8, eight-times-wide shared FFN), trained on 3B tokens of FineWeb-Edu to a final val loss of 2.82. It ran on a single A100 80GB on RunPod, using a warmup-stable-decay schedule with cosine decay over the last 300M tokens. About 40 hours of GPU time, roughly 394 US dollars, ~15K tok/s throughput. Intermediate snapshots at 1B and 2B tokens are published alongside the final checkpoint so anyone can probe M at mid-training as well as at the end.
- `mr-saoirse/orn-v2-108m` is a 108M parameter ORN V2 on 8B tokens of FineWeb-Edu. Trained on a single RTX 4090 for about 8 dollars. On standard benchmarks it scores 33.4% on HellaSwag, 62.4% on PIQA, and 46.2% on ARC-Easy. That beats GPT-2 Small at 27% fewer total parameters and roughly 48 times fewer coupling parameters.

Both are public. `orn predict --checkpoint orn-v3-605m` will fetch the 605M and generate from it; `orn diagnose` reads out M's spectral structure on whichever checkpoint you pass.

Frozen-M transfer: take a trained M, lock it, and train everything else from scratch on a different dataset. The resulting model recovers 98.6% of the validation loss you would have gotten training M from scratch. This is the cleanest confirmation that the coupling geometry is the invariant part. If M were encoding dataset-specific information, locking it would hurt. It does not.

There is also a small control task in the repo where the coupling rule is invariant by construction (a colour-matching task with periodic structure). I trained two four-layer models on it: one sharing a single A, B pair (MEMOISE) and one with a fresh pair at every layer (STORE). Both converge to similar loss. Coupling budget: 8,192 parameters for MEMOISE, 32,768 for STORE. Four times the compression, no meaningful quality loss.

![MEMOISE matches STORE on a task where coupling is provably invariant, at a quarter of the coupling parameters](https://raw.githubusercontent.com/Percolation-Labs/orn/article-drafts/drafts/figs/COM-01_colour_matching_compression.png)

## Why the point is not parameter saving

You can add up the arithmetic. At d=576, L=24, sharing coupling saves roughly fifteen million parameters. Redistribute those to width or depth and the model improves. That is real and it helps.

But the more interesting thing is what the shared M looks like once trained. In a 91M parameter ORN at d=512, the effective rank of M (90% of Frobenius energy) crystallises to about 20 within the first 200M tokens of training. Four percent of the full dimensionality carries the coupling rule. In the 108M V2 at d=576, the same measurement lands at 70 out of 576, still an order of magnitude below full dimension, with condition number around 10^7.

So not only is the coupling rule the same across layers, it is low-dimensional, and it is the same low-dimensional object across different models. Two transformers trained by different people on different data end up with layer-averaged M matrices whose eigenspectra correlate at 0.93+.

> If one M works at every layer of one model, and almost the same M works at every layer of a different model, then the model has discovered a shared geometry of which tokens couple to which, and that geometry is mostly a property of language, not of the training run.

This is what people mean when they talk about world models, minus the marketing. The eigenvectors of M form a basis for the coupling rule. Its rank is the dimension of the rule. Its spectrum is the rule's signature. We have not shown that this M is interpretable in the sense that you could read concepts off its eigenvectors. That is a harder open problem that our LORN programme is trying to address. What ORN shows is that the object exists, is reachable by a clean architectural move, and is stable across training runs.

Parameter saving is a bonus. The news is that there is something invariant in there to find.

## Running it

One pip install gets you the lot.

```
git clone https://github.com/Percolation-Labs/orn.git
cd orn
pip install -e .

# The motivating figure, reproduced locally in three seconds
orn reproduce COU-02 --save-plots figs/

# The full fast-tier suite of reproductions (fifteen seconds on a laptop)
orn reproduce SPE-01 COU-01 COU-02 COU-03 COM-01 OPE-01 OPE-04 LOR-01 \
              --save-plots figs/

# Pull the published 605M checkpoint and generate from it
pip install -e '.[data]'
orn predict --checkpoint orn-v3-605m --prompt "The key insight is"
orn diagnose --checkpoint orn-v3-605m
```

Every key result lives as a separate Python module that returns its headline numbers as a dict and, for most, emits the published figure as a PNG. A registry maps short codes (SPE-01, COU-02, COM-01) to modules. Pytest exercises every module as a smoke test. If a refactor breaks a claim, the test for that claim fails and the figure stops matching what the code produces.

The technical report, the published checkpoints, and the experimental harness are at [github.com/Percolation-Labs/orn](https://github.com/Percolation-Labs/orn). Feedback welcome.
