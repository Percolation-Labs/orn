"""
Spectral analysis of the SharedM coupling tensor.
===================================================

M = AB^T is the coupling geometry of the ORN. Its spectral structure reveals
what the model has learned about relational patterns:

- **Singular values** decompose M into independent coupling modes. The decay
  rate tells us how many dimensions are actually used for coupling.

- **Effective rank** (at 90% or 99% energy) is the key diagnostic. V1 at 91M
  converges to rank 19-20 out of 512 — only 20 dimensions encode English
  grammar's relational structure.

- **Condition number** (σ_max / σ_min) measures spectral spread. Kill criterion:
  must be > 2, otherwise M collapsed to a scaled identity.

- **Asymmetry** ||M - M^T|| / ||M|| measures how far M is from symmetric.
  Asymmetry is essential for autoregressive tasks (experimentally confirmed:
  symmetric LL^T fails, AB^T works).

- **Eigenvalues** (complex-valued for asymmetric M) are Koopman timescales.
  Real parts = convergence rates, imaginary parts = oscillation frequencies.
  V1 M has 89% complex eigenvalues — most coupling modes oscillate.

- **Spectral correlation** between two M matrices measures whether they share
  the same coupling structure (even if directions differ). Cross-layer
  correlation is 0.98 for pretrained transformers, motivating memoisation.
"""

from __future__ import annotations
import numpy as np
import torch
from typing import Optional


def spectral_analysis(M: np.ndarray, verbose: bool = False) -> dict:
    """
    Full spectral analysis of a coupling matrix M.

    Args:
        M: (d, d) coupling matrix (numpy array)
        verbose: Print results

    Returns:
        Dictionary with:
          - singular_values: Full SVD spectrum
          - eigenvalues: Complex eigenvalues
          - condition_number: σ_max / σ_min
          - effective_rank_90: Dims for 90% energy
          - effective_rank_99: Dims for 99% energy
          - spectral_gap: σ_1 / σ_2
          - asymmetry: ||M - M^T|| / ||M||
          - fraction_complex: Proportion of eigenvalues with nonzero imaginary part
          - energy_fractions: Per-mode energy fractions
    """
    sv = np.linalg.svd(M, compute_uv=False)
    eigvals = np.linalg.eigvals(M)

    energy = sv ** 2
    cumulative = np.cumsum(energy) / energy.sum()
    eff_rank_90 = int(np.searchsorted(cumulative, 0.9)) + 1
    eff_rank_99 = int(np.searchsorted(cumulative, 0.99)) + 1

    result = {
        "singular_values": sv,
        "eigenvalues": eigvals,
        "condition_number": float(sv[0] / sv[-1]) if sv[-1] > 1e-12 else float("inf"),
        "effective_rank_90": eff_rank_90,
        "effective_rank_99": eff_rank_99,
        "spectral_gap": float(sv[0] / sv[1]) if len(sv) > 1 and sv[1] > 1e-12 else float("inf"),
        "asymmetry": float(np.linalg.norm(M - M.T, "fro") / (np.linalg.norm(M, "fro") + 1e-10)),
        "fraction_complex": float(np.mean(np.abs(eigvals.imag) > 1e-10)),
        "energy_fractions": energy / energy.sum(),
    }

    if verbose:
        print(f"  Spectral Analysis of M ({M.shape[0]}×{M.shape[1]}):")
        print(f"    Effective rank (90%): {eff_rank_90}")
        print(f"    Effective rank (99%): {eff_rank_99}")
        print(f"    Condition number:     {result['condition_number']:.1f}")
        print(f"    Spectral gap (σ1/σ2): {result['spectral_gap']:.2f}")
        print(f"    Asymmetry:            {result['asymmetry']:.3f}")
        print(f"    Complex eigenvalues:  {result['fraction_complex']:.1%}")
        print(f"    Top 5 σ:              {sv[:5].round(3).tolist()}")

    return result


def spectral_analysis_from_model(model) -> dict:
    """
    Extract M = AB^T from a model and run spectral analysis.

    Works with any model that has .A and .B parameters (ORN, ORNV2, CORN).
    """
    A = model.A.detach().cpu().numpy()
    B = model.B.detach().cpu().numpy()
    M = A @ B.T
    return spectral_analysis(M)


def spectral_trajectory(checkpoints: list[dict]) -> dict:
    """
    Track M's spectral evolution across training checkpoints.

    Args:
        checkpoints: List of checkpoint dicts, each containing 'spectral' key
                     with 'eff_rank_90', 'condition_number', 'asymmetry', 'top_svs'
                     and optionally 'tokens_seen' or 'step'.

    Returns:
        Dictionary with arrays tracking spectral quantities over training:
          - steps: Step or token count at each checkpoint
          - ranks: Effective rank at each checkpoint
          - conditions: Condition number at each checkpoint
          - asymmetries: Asymmetry at each checkpoint
          - sv_trajectories: (n_checkpoints, n_svs) top singular values
    """
    steps = []
    ranks = []
    conditions = []
    asymmetries = []
    sv_trajectories = []

    for ckpt in checkpoints:
        spec = ckpt.get("spectral", {})
        steps.append(ckpt.get("tokens_seen", ckpt.get("step", 0)))
        ranks.append(spec.get("eff_rank_90", 0))
        conditions.append(spec.get("condition_number", 0))
        asymmetries.append(spec.get("asymmetry", 0))
        sv_trajectories.append(spec.get("top_svs", []))

    # Pad sv_trajectories to same length
    max_svs = max(len(sv) for sv in sv_trajectories) if sv_trajectories else 0
    sv_trajectories = [
        sv + [0.0] * (max_svs - len(sv)) for sv in sv_trajectories
    ]

    return {
        "steps": np.array(steps),
        "ranks": np.array(ranks),
        "conditions": np.array(conditions),
        "asymmetries": np.array(asymmetries),
        "sv_trajectories": np.array(sv_trajectories),
    }


def spectral_correlation(M1: np.ndarray, M2: np.ndarray) -> float:
    """
    Pearson correlation of sorted singular values between two matrices.

    This measures whether M1 and M2 share the same coupling *structure*,
    even if the directions (eigenvectors) differ. Cross-layer correlation
    is 0.98 for pretrained transformers — the structure IS invariant.

    Args:
        M1, M2: (d, d) coupling matrices

    Returns:
        Pearson correlation coefficient (float in [-1, 1])
    """
    sv1 = np.sort(np.linalg.svd(M1, compute_uv=False))[::-1]
    sv2 = np.sort(np.linalg.svd(M2, compute_uv=False))[::-1]

    # Truncate to same length
    n = min(len(sv1), len(sv2))
    sv1, sv2 = sv1[:n], sv2[:n]

    return float(np.corrcoef(sv1, sv2)[0, 1])


def correlation_length(sigma_1: float, sigma_2: float) -> float:
    """
    RG-inspired correlation length from top two singular values.

    ξ = 1 / ln(σ₁/σ₂)

    Large ξ means the top two modes are nearly degenerate — long-range
    correlations. Small ξ means rapid spectral decay — local coupling.
    """
    ratio = sigma_1 / (sigma_2 + 1e-10)
    return 1.0 / (np.log(ratio) + 1e-10)
