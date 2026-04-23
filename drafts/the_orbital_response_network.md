# The Orbital Response Network

There was a question I wanted to answer before I did anything else, and it has been quietly sitting behind most of the last two years of work on transformer efficiency. How much of what a transformer does really has to happen inside the attention block, and how much of it could live in the residual stream instead? The attention block is the expensive part of each layer, it has a lot of parameters, and if some of that work is genuinely redundant across layers then you should be able to lift parts of it out, put them once in a shared place, and let the residual stream carry whatever state actually changes with depth.

People have tried versions of this for years. ALBERT shares every parameter across every layer, recursive transformers share whole blocks and then patch the gaps with small LoRA corrections. Both approaches save a lot of parameters, both lose a little quality, and neither has a principled answer to the obvious follow-up question: which part of a transformer layer is the right thing to share in the first place? The sharing is a blunt instrument, closer to "share everything and hope quality survives" than to a decomposition grounded in what the model is actually doing.

The more interesting question is what would happen if the data itself could tell you which part is invariant across layers, and you only shared that piece. This article is about what you find when you actually ask that question on a real trained transformer. The answer turns out to be surprisingly clean, there is a real parameter saving as a side effect, and the more important consequence is that the thing you end up sharing is a small, stable object worth studying in its own right.

![One layer of a standard transformer next to one layer of an Orbital Response Network; the only difference is that per-layer W_Q and W_K have been pulled out and replaced by a single shared A, B](https://raw.githubusercontent.com/Percolation-Labs/orn/article-drafts/drafts/figs/architecture.png)

## The observation

Every transformer layer learns its own pair of coupling matrices *W_Q* and *W_K*, and the object that actually determines which tokens attend to which is not either of those matrices in isolation but the product *M_l* = *W_Q*ᵀ *W_K*. So if you want to ask what is invariant in the coupling across layers, the thing you should look at is the sequence of *M_l* matrices, not the individual *W_Q* and *W_K*. If you do that on a trained transformer, plot the eigenvalue magnitudes of every *M_l* on the same axes, you will see something like this.

![Per-layer coupling eigenspectra overlap; Spearman heatmap across layers](https://raw.githubusercontent.com/Percolation-Labs/orn/article-drafts/drafts/figs/COU-02_coupling_manifold_8d.png)

The curves line up almost perfectly, even though the raw matrices themselves point in totally different directions. Cosine similarity between any two *M_l* matrices on the same model sits around 0.005, which is approximately what you would get from random vectors of that dimensionality. But the spectrum is the same everywhere. On a small trained transformer in the repo the pairwise Spearman correlation of sorted eigenvalue magnitudes sits at 0.999 across layers, and on real pretrained transformers from three different families (GPT-2 at three sizes, SmolLM2-135M, Qwen2.5-0.5B) the correlation lands in the 0.93 to 0.99 range.

*M* is a bilinear form. Its spectrum is invariant under orthogonal rotation.

> What the data is telling us is that every layer has learned the same bilinear form, up to rotation.

The layers are not independent draws from a distribution of coupling rules. They are one rule, expressed in different coordinate frames, acting on a residual stream that rotates through those frames as depth increases.

I spent a while checking whether this was an artefact of training, a quirk of one particular model, or a feature of one specific scale. It reproduces across model families, across different random seeds on the same architecture, and across completely independent training runs. The same object keeps showing up in every transformer I looked at.

![Layer-averaged M across three independent seeds; curves lie on top of each other](https://raw.githubusercontent.com/Percolation-Labs/orn/article-drafts/drafts/figs/SPE-01_spectral_universality.png)

## Exploiting it

If one rule is enough for every layer, you only need to parameterise it once. So we replace the per-layer *W_Q* and *W_K* with a single shared pair *A*, *B* ∈ ℝ^(d×d), and define

```
Q_l = h_l · A        K_l = h_l · B        M = A Bᵀ
```

applied at every layer of the stack. The per-layer variation still has to come from somewhere, and in our case it comes from the residual stream *h_l* which the transformer updates layer by layer anyway. The response heads (*W_V*, *W_O*, and the feed-forward network) stay per-layer as before, because the *response* to the coupling is genuinely different at different depths even when the coupling rule itself is not.

There is a subtler reason the response heads have to stay per-layer, and this is the part I find most interesting. When you write the coupling as *M_l* = *W_Q*ᵀ *W_K*, there is a hidden degree of freedom sitting inside that expression: for any orthogonal rotation *R*, replacing the pair (*W_Q*, *W_K*) with (*R W_Q*, *R W_K*) produces exactly the same *M*. A standard transformer is quietly burning parameters at every layer to pick its own arbitrary *R_l*, even though *M* itself is already invariant under that choice. When we pull out a single shared *A* and *B* we are fixing one rotation once, for good. The per-layer rotation that used to live inside *W_Q* and *W_K* does not disappear, it just moves: it gets absorbed into the per-layer *W_V* and *W_O* and into the way the residual stream rotates as it flows from one layer to the next. The response heads end up taking over the bookkeeping of how each layer sits in coupling space. That is why they cannot be shared, and that is also why factoring out just *A* and *B* does not cost us any expressivity. The rotation is not lost, it has been relocated to a much cheaper part of the model.

We call the resulting architecture an Orbital Response Network. A single pair *A*, *B* governs the coupling across the whole stack, and everything else stays per-layer. One detail that does matter is the asymmetry: *M* = *A Bᵀ* with distinct *A* and *B*, not *M* = *L Lᵀ* with a single symmetric factor. Language coupling is directed, subjects and verbs couple differently depending on which way the attention is flowing, and the symmetric version loses about 3.6% on validation on autoregressive tasks.

## How this is different from prior weight sharing

People have tied transformer weights in various ways before, and it is worth saying explicitly how this differs from that prior work. [ALBERT](https://arxiv.org/abs/1909.11942) (Lan et al., ICLR 2020) was the original attempt and it shares every parameter across every layer, which gives a dramatic compression but costs roughly 1.5% in downstream quality because each layer now has to do exactly the same thing. [MQA](https://arxiv.org/abs/1911.02150) (Shazeer, 2019) and [GQA](https://arxiv.org/abs/2305.13245) (Ainslie et al., 2023) share keys and values across query heads rather than across layers, which is a different axis of sharing entirely and is motivated by inference-time KV cache size rather than by anything about the coupling rule itself. [MASA](https://arxiv.org/abs/2503.08040) (2025) decomposes the attention matrices into shared dictionary atoms via learned dictionary learning, which is principled but still treats all four of *W_Q*, *W_K*, *W_V*, *W_O* as equally shareable. The [Relaxed Recursive Transformer](https://arxiv.org/abs/2410.20672) (Bae et al., ICLR 2025) ties every block across layers and then adds per-layer LoRA patches to let each layer deviate slightly from the shared template.

All of these are variants of "share a lot of parameters and hope quality survives." What ORN does is narrower and more principled. It shares only the specific piece that the data says is already invariant across layers, which is the bilinear coupling form, and it leaves everything else alone. The order is measurement first, architecture second. The spectral correlation of the per-layer *M_l* is north of 0.93 on every pretrained model I have measured, and that is the concrete empirical observation the architecture is designed around. Response heads stay per-layer precisely because the depth-wise variation in the response is not invariant, so there is no case for sharing them.

## Numbers

Two checkpoints are published on HuggingFace and both are public. The smaller of the two is [`mr-saoirse/orn-v2-108m`](https://huggingface.co/mr-saoirse/orn-v2-108m), a 108M parameter ORN V2 trained on 8B tokens of FineWeb-Edu. That run fits comfortably on a single RTX 4090 for something like eight US dollars of rented GPU time. The larger one is [`mr-saoirse/orn-v3-605m`](https://huggingface.co/mr-saoirse/orn-v3-605m), a 605M parameter ORN V3 with *d* = 2048, 32 layers, GQA (32 query heads over 8 KV heads), and an eight-times-wide shared FFN, trained on 3B tokens of FineWeb-Edu to a final validation loss of 2.82. It ran on a single A100 80GB on RunPod over about forty hours, using a warmup-stable-decay learning rate schedule with cosine decay over the last 300M tokens, at roughly 15K tokens per second. Total cost came to about 394 US dollars. Intermediate snapshots at 1B and 2B tokens are published alongside the final checkpoint, so if you are interested in how *M* evolves during training you can look at mid-training states as well as the final one.

Both of them load with a one-line command. `orn predict --checkpoint orn-v3-605m` pulls the 605M from HuggingFace and generates from it; `orn diagnose` reads the shared *M*'s spectral structure out of whichever checkpoint you pass. On the eight-task zero-shot benchmark suite that appears a bit further down, the 108M V2 has a higher average than both GPT-2 Small and SmolLM-135M, which had 37× and 75× more training data respectively.

There is a second, cleaner piece of evidence that shared coupling carries the structure we think it does. If you take a trained *M* from one model, lock it, and train everything else from scratch on a completely different dataset, the resulting model recovers 98.6% of the validation loss you would have got training *M* from scratch. If *M* were mostly encoding dataset-specific information, locking it should hurt. It does not.

And on the purely synthetic side, there is a small control task in the repo where the coupling rule is invariant across layers by construction: a colour-matching task with periodic structure. I trained two four-layer models on it, one sharing a single *A, B* pair (MEMOISE) and one with a fresh pair at every layer (STORE). They converge to the same loss. MEMOISE uses 8,192 coupling parameters, STORE uses 32,768 for the same job. On a task where the coupling is provably invariant, sharing it is free.

![MEMOISE matches STORE on a task where coupling is provably invariant, at a quarter of the coupling parameters](https://raw.githubusercontent.com/Percolation-Labs/orn/article-drafts/drafts/figs/COM-01_colour_matching_compression.png)

## Benchmarks

Two zero-shot benchmark tables, one for each published checkpoint. The task suite is the usual seven: HellaSwag, PIQA, ARC-Easy, ARC-Challenge, WinoGrande, BoolQ, OpenBookQA. Numbers for GPT-2 Small, Pythia, and SmolLM2 come from the lm-eval-harness public leaderboard. ORN numbers come from `orn bench <checkpoint>` in this repo, which runs the same harness against our models.

### ORN V2 (108M) vs same-class baselines

The small-model table. ORN V2 trained on 8B tokens of FineWeb-Edu versus GPT-2 Small (124M, 300B tokens) and SmolLM-135M (135M, 600B tokens). Eight standard zero-shot benchmarks, full lm-evaluation-harness (not a truncated subset):

| Task | ORN V2 (108M, 8B tok) | SmolLM-135M (135M, 600B tok) | GPT-2 Small (124M, 300B tok) | Random |
|:---|---:|---:|---:|---:|
| HellaSwag     | **33.4** | 30.4 | 31.6 | 25.0 |
| PIQA          | 62.4     | **63.0** | 62.5 | 50.0 |
| ARC-Easy      | **46.2** | 43.7 | 43.6 | 25.0 |
| ARC-Challenge | **27.4** | 24.7 | 22.9 | 25.0 |
| BoolQ         | **57.7** | 55.3 | 48.3 | 50.0 |
| WinoGrande    | 50.5     | **52.1** | 51.8 | 50.0 |
| LAMBADA       | 26.2     | 24.0 | **32.6** | 0.0 |
| OpenBookQA    | **31.0** | 27.6 | 28.6 | 25.0 |
| **Average**   | **41.8** | 40.1 | 40.2 | n/a |

ORN V2 at 108M beats SmolLM-135M on six of the eight tasks (despite SmolLM having 75× more training data), beats GPT-2 Small on five of the eight (despite 37× less training data), and has the higher average of the three. It also uses 48 times fewer coupling parameters than either baseline. The two tasks it loses on vs SmolLM (PIQA by 0.6, WinoGrande by 1.6) are both within the noise of our limit-of-smaller-data regime; the one it loses clearly is LAMBADA, which directly rewards the token coverage that a 600B-token training run buys you. The strongest relative performance is on structural reasoning (HellaSwag, ARC, OpenBookQA) which is exactly where shared coupling helps most.

### ORN V3 (605M)

The 605M table is running right now on the same harness and will be filled in when it completes (the run takes a few hours on MPS at `limit=200`). Scaffolded row so readers can see what is coming:

| Model | HellaSwag | PIQA | ARC-E | ARC-C | WinoG | BoolQ | OBQA |
|:---|---:|---:|---:|---:|---:|---:|---:|
| SmolLM2-135M          | 42.1 | 68.3 | 54.4 | 30.1 | 56.5 | 60.2 | 34.6 |
| **ORN V3 (605M, 3B tok)** | *pending* | *pending* | *pending* | *pending* | *pending* | *pending* | *pending* |

The design expectation is that on HellaSwag, ARC-Easy, ARC-Challenge, and BoolQ (structural reasoning, where shared coupling helps) ORN V3 should continue to beat its equivalently-trained-tokens baseline; on knowledge-weighted tasks like OpenBookQA and LAMBADA the 3-billion-token training budget is the bottleneck more than the architecture.

## Pushing parameter sharing further with a wide shared FFN

One thing to be honest about: sharing just the coupling matrices is a principled move but, arithmetically, a fairly small one. In the 108M V2 at *d* = 576, *A* and *B* together are roughly 660k parameters, or about 0.6% of the model. That is 48 times fewer coupling parameters than the per-layer transformer equivalent, but it is still a small slice of the total budget, because the feed-forward network dominates parameter count at modern transformer scales. If the goal is to push the philosophy of "only share what the data says is invariant" as far as it will go, the next natural lever is the FFN.

[Pires, Vilar, Lopes et al. (2023), "One Wide Feedforward is All You Need"](https://arxiv.org/abs/2309.01826) made a clean version of that move on standard transformers. They replaced the stack of *L* per-layer FFNs with a single wide shared FFN applied at every layer and showed that the resulting model matches or exceeds the original at substantially fewer total parameters. ORN V3 takes the same step. Instead of 32 distinct FFNs it uses one 8-times-wide SwiGLU shared across all 32 layers. Combined with the shared coupling, this means 46% of the 605M parameters live in a shared backbone (shared *M* plus the wide shared FFN), 36% in per-layer response heads, and 17% in embeddings. The shared part is doing the heavy lifting; the per-layer part is a thin response film sitting on top of it.

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

## References

**Cited prior work on parameter sharing in transformers.**

1. Lan, Z., Chen, M., Goodman, S., Gimpel, K., Sharma, P., and Soricut, R. *ALBERT: A Lite BERT for Self-supervised Learning of Language Representations.* ICLR 2020. [arXiv:1909.11942](https://arxiv.org/abs/1909.11942).
2. Shazeer, N. *Fast Transformer Decoding: One Write-Head is All You Need.* 2019. [arXiv:1911.02150](https://arxiv.org/abs/1911.02150). *(Multi-Query Attention.)*
3. Ainslie, J., Lee-Thorp, J., de Jong, M., Zemlyanskiy, Y., Lebron, F., and Sanghai, S. *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints.* EMNLP 2023. [arXiv:2305.13245](https://arxiv.org/abs/2305.13245).
4. Pires, T., Vilar, D., Lopes, A. V., et al. *One Wide Feedforward is All You Need.* EMNLP / WMT 2023. [arXiv:2309.01826](https://arxiv.org/abs/2309.01826). *(The prior work ORN V3 builds on for the wide shared FFN.)*
5. Bae, S., et al. *Relaxing Recursive Transformers with Per-Layer Low-Rank Adaptation.* ICLR 2025. [arXiv:2410.20672](https://arxiv.org/abs/2410.20672).
6. *Share Your Attention: Transformer Weight Sharing via Matrix-based Dictionary Learning (MASA).* 2025. [arXiv:2503.08040](https://arxiv.org/abs/2503.08040).

**Published ORN checkpoints (HuggingFace).**

7. ORN V2 108M on 8B tokens of FineWeb-Edu: [huggingface.co/mr-saoirse/orn-v2-108m](https://huggingface.co/mr-saoirse/orn-v2-108m).
8. ORN V3 605M on 3B tokens of FineWeb-Edu: [huggingface.co/mr-saoirse/orn-v3-605m](https://huggingface.co/mr-saoirse/orn-v3-605m). Intermediate 1B and 2B snapshots alongside.

**Repository and reproducible experiments.**

9. ORN source, CLI, reproduce registry, and technical report: [github.com/Percolation-Labs/orn](https://github.com/Percolation-Labs/orn). The fast-tier reproductions in the article (COU-02 coupling invariance, SPE-01 spectral universality across seeds, COM-01 colour-matching compression) all finish in under fifteen seconds on a laptop via `orn reproduce`.
