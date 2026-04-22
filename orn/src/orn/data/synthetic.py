"""
Synthetic tasks for controlled ORN experiments.
=================================================

These tasks are designed to isolate the memoisation hypothesis: they have
coupling rules that are provably invariant across layers, so a shared M
SHOULD work. If it doesn't, the architecture is broken.

Color matching (Experiment 1):
  - N tokens, each with a color label and content vector
  - Target = mean of same-colored tokens' content vectors
  - Coupling rule: "same color → high affinity" (identical at every layer)
  - Result: SharedM matches per-layer with 16x fewer coupling params

Pattern sequences (Experiment 2):
  - Hierarchical patterns: base motif + modifiers + noise
  - Richer coupling structure requiring compositional attention
  - Used for scaling experiments (loss vs. parameter budget)
"""

from __future__ import annotations
import torch
import numpy as np


def make_color_matching_batch(
    batch_size: int,
    seq_len: int = 48,
    d_model: int = 64,
    num_colors: int = 16,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate a color matching task batch.

    Each token has a random color and a random content vector. The target
    for each token is the mean content vector of all tokens with the same
    color. This requires learning "same color → attend" — a pure coupling
    task with invariant structure.

    Args:
        batch_size: Number of sequences
        seq_len: Tokens per sequence
        d_model: Content vector dimension
        num_colors: Number of distinct colors
        device: Target device

    Returns:
        colors: (B, S) color labels (int)
        tokens: (B, S, D) content vectors
        targets: (B, S, D) mean of same-colored tokens
    """
    device = torch.device(device) if isinstance(device, str) else device

    colors = torch.randint(0, num_colors, (batch_size, seq_len))
    tokens = torch.randn(batch_size, seq_len, d_model)
    targets = torch.zeros_like(tokens)

    for c in range(num_colors):
        mask = (colors == c).unsqueeze(-1).float()
        counts = mask.sum(dim=1, keepdim=True).clamp(min=1)
        color_mean = (tokens * mask).sum(dim=1, keepdim=True) / counts
        targets += color_mean * mask

    return (
        colors.to(device),
        tokens.to(device),
        targets.to(device),
    )


def make_pattern_batch(
    batch_size: int,
    seq_len: int = 64,
    num_tokens: int = 128,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate hierarchical pattern sequences for scaling experiments.

    Sequences contain:
    1. A base repeating motif (period 4-8)
    2. Additive modifiers at regular intervals
    3. Random noise tokens

    The model must learn multi-scale coupling: local (within-motif),
    medium-range (motif-to-motif), and global (modifier patterns).

    Args:
        batch_size: Number of sequences
        seq_len: Tokens per sequence
        num_tokens: Vocabulary size
        device: Target device

    Returns:
        input_ids: (B, S) input token IDs
        target_ids: (B, S) shifted target token IDs
    """
    device = torch.device(device) if isinstance(device, str) else device
    seq = torch.zeros(batch_size, seq_len + 1, dtype=torch.long)

    for b in range(batch_size):
        period = torch.randint(4, 9, (1,)).item()
        motif = torch.randint(0, num_tokens, (period,))

        # Base pattern
        for i in range(seq_len + 1):
            seq[b, i] = motif[i % period]

        # Additive modifier every 12-16 positions
        mod_period = torch.randint(12, 17, (1,)).item()
        mod_offset = torch.randint(0, num_tokens, (1,)).item()
        for i in range(0, seq_len + 1, mod_period):
            seq[b, i] = (seq[b, i] + mod_offset) % num_tokens

        # 10% noise
        noise_mask = torch.rand(seq_len + 1) < 0.1
        seq[b, noise_mask] = torch.randint(0, num_tokens, (noise_mask.sum(),))

    x = seq[:, :-1].to(device)
    y = seq[:, 1:].to(device)
    return x, y
