"""SPE-03 — the real motivating figure: W_Q^T W_K eigenspectra across layers
of pretrained transformers.

This is the result that motivated shared M. We pull a pretrained transformer
from HuggingFace (default: gpt2, ~120MB), extract each layer's W_Q^T W_K,
compute sorted |eigenvalue| curves, and measure their Spearman correlation
across layers. The published finding is mean ρ ≥ 0.93 across GPT-2 / SmolLM2 /
Qwen2.5 even when raw matrices have cosine ~0.01.

Requires `pip install -e '.[data]'` for transformers.
Runtime: ~1-2 minutes including the model download on first use.

Use:
    orn reproduce SPE-03 --save-plots figs/ --slow
    orn reproduce SPE-03 --slow               # numbers only
"""
from __future__ import annotations

import numpy as np
import torch

from orn.diagnostics.plots import plot_layer_invariance


def _extract_qk_per_layer(model_name: str):
    """Pull W_Q and W_K per layer from a HuggingFace causal LM."""
    from transformers import AutoModelForCausalLM  # type: ignore

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    model.eval()
    per_layer_M: list[np.ndarray] = []

    # GPT-2 style: transformer.h[i].attn.c_attn holds stacked [q, k, v] weights.
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        for block in model.transformer.h:
            W = block.attn.c_attn.weight.detach().double().numpy()   # (3d, d)
            d = W.shape[1]
            Wq = W[:d].T       # (d, d)  — c_attn weight is (out_features, in_features)
            Wk = W[d:2*d].T
            per_layer_M.append(Wq.T @ Wk)
        return per_layer_M

    # LLaMA-style: each block has self_attn.{q_proj,k_proj}.
    try:
        layers = model.model.layers  # llama, qwen, smollm
    except AttributeError as e:
        raise RuntimeError(f"Don't know how to extract Q/K from {type(model).__name__}") from e

    for block in layers:
        Wq = block.self_attn.q_proj.weight.detach().double().numpy()  # (hq*dh, d)
        Wk = block.self_attn.k_proj.weight.detach().double().numpy()  # (hk*dh, d)
        d = Wq.shape[1]
        # Average over heads (each block has n_q_heads groups of dh rows). Simpler:
        # treat the full flattened map — we just need the spectrum of W_Q^T W_K.
        per_layer_M.append(Wq.T @ Wk[:Wq.shape[0]])  # aligned on GQA
    return per_layer_M


def run(device: str | None = None, model_name: str = "gpt2") -> dict:
    try:
        from scipy.stats import spearmanr
    except ImportError as e:
        raise ImportError("scipy is required") from e

    with np.errstate(all="ignore"):
        per_layer_M = _extract_qk_per_layer(model_name)
        eigs = [np.sort(np.abs(np.linalg.eigvals(M.astype(np.float64))))[::-1]
                for M in per_layer_M]
        d = min(len(e) for e in eigs)
        eigs_trim = [e[:d] for e in eigs]
        L = len(eigs_trim)
        corrs = []
        for i in range(L):
            for j in range(i + 1, L):
                rho, _ = spearmanr(eigs_trim[i], eigs_trim[j])
                corrs.append(float(rho))

    return {
        "model":                   model_name,
        "n_layers":                L,
        "mean_pairwise_spearman":  float(np.mean(corrs)),
        "min_pairwise_spearman":   float(np.min(corrs)),
        "_per_layer_M":            per_layer_M,
    }


def plot(result: dict, save_path=None):
    Ms = result["_per_layer_M"]
    return plot_layer_invariance(
        Ms, titles=[f"L{i}" for i in range(len(Ms))],
        save_path=save_path,
        suptitle=f"{result['model']}: per-layer W_Q^T W_K "
                 f"(mean ρ = {result['mean_pairwise_spearman']:.3f})",
    )
