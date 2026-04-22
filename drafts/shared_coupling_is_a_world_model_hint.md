# A shared coupling rule, not a shared block: what a transformer has in common with itself

I want to describe what I think the most interesting thing in a trained transformer is. Most people who talk about transformer redundancy talk about it as a compression opportunity. I think that is a small version of the story.

The larger version goes like this. Every transformer layer computes an attention step with its own pair of matrices W_Q and W_K. People have noticed for years that these are wasteful somehow, and there are papers going back to ALBERT that tie them together to save parameters. That always loses a little quality, and nobody has a principled answer for which part to share.

The principled answer falls out of one measurement. Take a trained transformer. At each layer, form M_l = W_Q^T W_K. This is the effective coupling matrix, the thing inside the attention softmax that says which tokens couple to which. Sort its eigenvalue magnitudes. Plot the curves on top of each other.

![Per-layer eigenspectra lining up, Spearman correlation heatmap](https://raw.githubusercontent.com/Percolation-Labs/orn/article-drafts/drafts/figs/COU-02_coupling_manifold_8d.png)

They line up. The raw matrices point in totally different directions, cosine similarity somewhere around 0.005 between any two layers. But the spectrum is the same everywhere. On a small ORN model the correlation is 0.999. On actual pretrained transformers (GPT-2 across three sizes, SmolLM2, Qwen2.5), the pairwise Spearman sits between 0.93 and 0.99.

Read this carefully. The coupling matrix W_Q^T W_K is a bilinear form. Its spectral content is invariant under orthogonal rotation. What the data says is that every layer has learned the same bilinear form up to rotation. The layers are not independent draws from some distribution of coupling rules. They are the same rule in different coordinate frames, acting on a residual stream that rotates through those frames as depth increases.

## How to use this

If one rule is enough for every layer, you should parameterise it once. Replace per-layer W_Q and W_K with a single shared pair A, B in R^{d x d}. Define M = A B^T. At every layer:

```
Q_l = h_l · A       K_l = h_l · B
attn_l = softmax(Q_l K_l^T / sqrt(d_h)) V_l
```

Per-layer variation comes from the residual stream h_l, which the transformer already updates layer by layer. Per-layer response heads (W_V, W_O, FFN) stay per-layer because the response to coupling legitimately varies with depth. The coupling geometry does not.

This is what an Orbital Response Network is.

The asymmetry matters. M = A B^T with A not equal to B, not M = L L^T. Language coupling is directed (subject and verb couple differently depending on which is attending to which), and symmetric coupling fails on autoregressive tasks by about 3.6% on held-out loss.

## Numbers

Published checkpoints are on HuggingFace at `mr-saoirse/orn-v3-605m` and `mr-saoirse/orn-v2-108m`, both public. The 605M ORN V3 (d=2048, L=32, 8x wide shared FFN) is the largest published run, trained on 2B tokens of FineWeb-Edu, and is what you get by default if you run `orn predict --checkpoint orn-v3-605m` from the repo. The 108M V2 on 7B tokens scores 33.4% on HellaSwag, 62.4% on PIQA, 46.2% on ARC-Easy, beating GPT-2 Small on matched tasks at 27% fewer total parameters and roughly 48 times fewer coupling parameters.

Frozen-M transfer: take M from a trained model, lock it, train everything else from scratch on a different dataset. The resulting model recovers 98.6% of the validation loss you would have gotten training M from scratch. This is the cleanest confirmation that the coupling geometry is the invariant part.

The trained M is also low-dimensional in a specific way. In a 91M parameter ORN at d=512, the effective rank (90 percent of Frobenius energy) crystallises to about 20 during training. Four percent of the full dimensionality carries the coupling rule. In a 108M V2 at d=576, the same measurement lands at 70 out of 576, still an order of magnitude below full dimension, with condition number order 10^7 indicating sharp spectral structure.

## Why this is not really a compression story

You can add up the arithmetic. At d=576, L=24, sharing coupling saves roughly 15M parameters. You can spend those on width or depth and get a better model. That is real and nice but it is the least interesting consequence.

The interesting consequence is that the model has collapsed a space of per-layer coupling rules into one object. That object has low rank. Its eigenvectors are stable under training. Its spectrum is universal across model families. Two different pretrained transformers, trained on different data by different teams, end up with layer-averaged coupling M's that correlate at 0.93+ in eigenspectrum.

That is not a compression story. That is the same object showing up twice.

> If one M works at every layer of one model, and almost the same M works at every layer of a different model, then the model has discovered a shared geometry of which tokens couple to which, and that geometry is mostly a property of the data, not of the training run.

This is what people mean when they talk about world models, minus the marketing. The thing that is invariant across layers and across training runs is the rule by which language composes. You can read the eigenvectors of M as a basis for that rule. The rank of M is the dimension of the rule. The spectral structure is its signature.

I do not think we have yet shown that this M is a *useful* world model, in the sense that you can navigate it, intervene on it, or read concepts off its eigenvectors in an interpretable way. That is what the adjacent LORN programme is trying to do and it is a harder, unfinished problem. What ORN shows is that the object exists and is reachable by a clean architectural move.

## Running it

Everything below runs with a single pip install.

```
git clone https://github.com/Percolation-Labs/orn.git
cd orn
pip install -e .

# The motivating figure
orn reproduce COU-02 --save-plots figs/

# The full fast-tier suite of reproductions (about fifteen seconds)
orn reproduce SPE-01 COU-01 COU-02 COU-03 COM-01 OPE-01 OPE-04 LOR-01 \
              --save-plots figs/

# Pull the published 605M checkpoint and generate from it
pip install -e '.[data]'
orn predict --checkpoint orn-v3-605m --prompt "The key insight is"
orn diagnose --checkpoint orn-v3-605m
```

Each result in the reproduce suite is a separate Python module that returns its headline numbers as a dict and, for most, emits the paper figure as a PNG. A registry maps short codes (SPE-01, COU-01, COM-01) to the modules. If a refactor breaks a claim, the pytest case for that claim fails, and the figure in the readme stops matching what the code produces. That is the discipline I am trying to hold the repo to.

The technical report is at [github.com/Percolation-Labs/orn](https://github.com/Percolation-Labs/orn). The published checkpoints are the 605M V3 and 108M V2 listed above. Feedback welcome.
