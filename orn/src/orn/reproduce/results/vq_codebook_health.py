"""LOR-01 — the VQREncoder codebook fills and does not collapse.

Train a tiny LORN on synthetic "multiset" inputs (each sequence is a random
bag of K tokens drawn from a class-specific distribution) for a small number
of steps, then report codebook-occupancy diagnostics:

  codes_used          : how many of K codebook entries have ever been chosen
  max_cluster_frac    : fraction of batch examples in the largest cluster
  cluster_entropy     : Shannon entropy of the cluster distribution (bits)

Healthy signal:
  codes_used climbing toward K (no collapse).
  max_cluster_frac dropping below ~2/K (balanced usage).
  cluster_entropy rising from ~0 to close to log2(K).
"""
from __future__ import annotations

import math

import torch

from orn.models.lorn import LORN, LORNConfig


def _class_batch(batch_size: int = 16, n_classes: int = 4, K: int = 8, vocab: int = 32):
    """Each row is a random permutation drawn from one of n_classes class vocabs."""
    rows, labels = [], []
    per_class = [torch.randperm(vocab)[:K] for _ in range(n_classes)]
    for _ in range(batch_size):
        c = torch.randint(0, n_classes, (1,)).item()
        row = per_class[c][torch.randperm(K)]
        rows.append(row)
        labels.append(c)
    return torch.stack(rows), torch.tensor(labels)


def run(device: str | None = None, train_steps: int = 400, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    cfg = LORNConfig(
        vocab_size=32, T_max=8,
        d_r=32, r_depth=2, r_heads=2, codebook_size=4, entropy_weight=0.5,
        d_l=64, l_layers=2, l_q_heads=4, l_kv_heads=2, l_d_head=16,
        l_ffn_width_mult=1, bridge_rank=2, n_inject_layers=2,
    )
    m = LORN(cfg)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)

    codes_seen: set[int] = set()

    for step in range(train_steps):
        x, _labels = _class_batch(batch_size=16, n_classes=cfg.codebook_size,
                                   K=cfg.T_max, vocab=cfg.vocab_size)
        y = torch.roll(x, -1, dims=1)
        alpha, aux = m.R(x)
        # Update codebook via straight-through loss:
        vq = m.R.vq_loss(aux)
        opt.zero_grad(); vq.backward(); opt.step()

        chosen = aux["chosen"].cpu().tolist()
        codes_seen.update(chosen)

    # Final diagnostics on a fresh probe batch
    with torch.no_grad():
        x, _ = _class_batch(batch_size=64)
        _, aux = m.R(x)
        chosen = aux["chosen"]
        counts = torch.bincount(chosen, minlength=cfg.codebook_size).float()
        probs = counts / counts.sum()
        max_frac = probs.max().item()
        entropy = -(probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log2()).sum().item()

    return {
        "codebook_size":    cfg.codebook_size,
        "codes_used":       len(codes_seen),
        "max_cluster_frac": max_frac,
        "cluster_entropy":  entropy,
        "max_entropy":      math.log2(cfg.codebook_size),
    }
