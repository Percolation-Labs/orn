"""
Visualization utilities for ORN diagnostics.
==============================================

Standard plots for tracking M's spectral evolution, comparing models,
and generating training reports. All plots use the Agg backend for
non-interactive environments (Vast.ai, CI).
"""

from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from typing import Optional

from orn.diagnostics.spectral import spectral_analysis


def plot_spectrum(M: np.ndarray, title: str = "M Spectral Structure",
                  save_path: Optional[str | Path] = None) -> plt.Figure:
    """
    Plot M's singular value spectrum and eigenvalue distribution.

    Two panels:
    1. Singular values (linear + log scale) with effective rank markers
    2. Eigenvalue scatter in the complex plane (Re vs Im)

    Args:
        M: (d, d) coupling matrix
        title: Plot title
        save_path: Optional path to save PNG
    """
    spec = spectral_analysis(M)
    sv = spec["singular_values"]
    eigvals = spec["eigenvalues"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Panel 1: Singular values (linear)
    ax = axes[0]
    ax.bar(range(len(sv)), sv, color="steelblue", alpha=0.8)
    ax.axvline(spec["effective_rank_90"] - 0.5, color="red", ls="--", label=f"Rank-90: {spec['effective_rank_90']}")
    ax.axvline(spec["effective_rank_99"] - 0.5, color="orange", ls="--", label=f"Rank-99: {spec['effective_rank_99']}")
    ax.set_xlabel("Mode index")
    ax.set_ylabel("Singular value")
    ax.set_title("Singular Values")
    ax.legend(fontsize=8)

    # Panel 2: Singular values (log)
    ax = axes[1]
    ax.semilogy(sv, "o-", color="steelblue", markersize=3)
    ax.set_xlabel("Mode index")
    ax.set_ylabel("Singular value (log)")
    ax.set_title(f"Log Spectrum (κ = {spec['condition_number']:.0f})")
    ax.grid(True, alpha=0.3)

    # Panel 3: Eigenvalue scatter
    ax = axes[2]
    ax.scatter(eigvals.real, eigvals.imag, s=15, c="steelblue", alpha=0.7)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.set_xlabel("Re(λ)")
    ax.set_ylabel("Im(λ)")
    ax.set_title(f"Eigenvalues ({spec['fraction_complex']:.0%} complex)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_spectral_trajectory(trajectory: dict, title: str = "M Crystallisation",
                             save_path: Optional[str | Path] = None) -> plt.Figure:
    """
    Plot M's spectral evolution across training checkpoints.

    Shows how effective rank, condition number, and top singular values
    change during training. The key signature is "crystallisation" — rank
    drops and stabilises as M finds the true coupling dimensionality.

    Args:
        trajectory: Output of spectral_trajectory()
        title: Plot title
        save_path: Optional path to save PNG
    """
    steps = trajectory["steps"]
    x_label = "Tokens" if steps.max() > 10000 else "Steps"

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Effective rank
    ax = axes[0, 0]
    ax.plot(steps, trajectory["ranks"], "o-", color="steelblue")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Effective Rank (90%)")
    ax.set_title("M Crystallisation")
    ax.grid(True, alpha=0.3)

    # Condition number
    ax = axes[0, 1]
    ax.semilogy(steps, trajectory["conditions"], "o-", color="darkorange")
    ax.axhline(2, color="red", ls="--", alpha=0.5, label="Kill: κ < 2")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Condition Number (log)")
    ax.set_title("Spectral Spread")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Asymmetry
    ax = axes[1, 0]
    ax.plot(steps, trajectory["asymmetries"], "o-", color="green")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Asymmetry ||M-M^T||/||M||")
    ax.set_title("Asymmetry (should be > 0)")
    ax.grid(True, alpha=0.3)

    # Top singular value trajectories
    ax = axes[1, 1]
    svt = trajectory["sv_trajectories"]
    n_svs = min(5, svt.shape[1])
    for i in range(n_svs):
        ax.plot(steps, svt[:, i], "o-", label=f"σ_{i+1}", markersize=4)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Singular Value")
    ax.set_title("Top Singular Values")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_attention(alpha: np.ndarray, labels: Optional[list] = None,
                   title: str = "Attention Weights",
                   save_path: Optional[str | Path] = None) -> plt.Figure:
    """
    Heatmap of attention weights with annotations.

    Args:
        alpha: (N, N) attention weight matrix
        labels: Token labels
        title: Plot title
        save_path: Optional path to save PNG
    """
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    im = ax.imshow(alpha, cmap="Blues", vmin=0, vmax=1)

    N = alpha.shape[0]
    if labels is None:
        labels = [str(i) for i in range(N)]

    ax.set_xticks(range(N))
    ax.set_xticklabels(labels, fontsize=8, rotation=45, ha="right")
    ax.set_yticks(range(N))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Key")
    ax.set_ylabel("Query")
    ax.set_title(title)

    for i in range(N):
        for j in range(N):
            color = "white" if alpha[i, j] > 0.5 else "black"
            ax.text(j, i, f"{alpha[i,j]:.2f}", ha="center", va="center",
                    color=color, fontsize=7)

    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_scaling_curves(results: dict, title: str = "Scaling Comparison",
                        save_path: Optional[str | Path] = None) -> plt.Figure:
    """
    Plot scaling curves: loss vs parameters for ORN and transformer.

    Args:
        results: Dict with keys like "orn" and "transformer", each mapping to
                 a dict with "params" (list) and "losses" (list).
        title: Plot title
        save_path: Optional path to save PNG
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    colors = {"orn": "steelblue", "transformer": "darkorange", "corn": "green"}

    # Log-log scaling plot
    ax = axes[0]
    for name, data in results.items():
        params = np.array(data["params"])
        losses = np.array(data["losses"])
        color = colors.get(name, "gray")
        ax.loglog(params, losses, "o-", label=name.upper(), color=color, markersize=6)
    ax.set_xlabel("Total Parameters")
    ax.set_ylabel("Final Loss")
    ax.set_title("Scaling Curves (log-log)")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")

    # Coupling parameter comparison
    ax = axes[1]
    for name, data in results.items():
        if "coupling_params" in data:
            params = np.array(data["params"])
            coupling = np.array(data["coupling_params"])
            color = colors.get(name, "gray")
            ax.bar(
                np.arange(len(params)) + (0.35 if name != "orn" else 0),
                coupling / params * 100,
                0.35, label=f"{name.upper()} coupling %",
                color=color, alpha=0.7,
            )
    ax.set_xlabel("Config index")
    ax.set_ylabel("Coupling params (%)")
    ax.set_title("Parameter Budget: Coupling")
    ax.legend(fontsize=8)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_training_report(log_entries: list[dict], model_name: str = "ORN",
                         save_path: Optional[str | Path] = None) -> plt.Figure:
    """
    Generate a comprehensive training report from JSONL log entries.

    Four panels: train/val loss, learning rate, M rank, M condition number.
    This is the standard report generated at every checkpoint.

    Args:
        log_entries: List of dicts from training log (each with step, train_loss,
                     val_loss, lr, M_rank, M_cond, etc.)
        model_name: Name for the title
        save_path: Optional path to save PNG
    """
    steps = [e["step"] for e in log_entries]
    train_loss = [e.get("train_loss", None) for e in log_entries]
    val_loss = [e.get("val_loss", None) for e in log_entries]
    lr = [e.get("lr", None) for e in log_entries]
    m_rank = [e.get("M_rank", None) for e in log_entries]
    m_cond = [e.get("M_cond", None) for e in log_entries]

    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    # Train/Val loss
    ax = fig.add_subplot(gs[0, 0])
    if any(v is not None for v in train_loss):
        ax.plot(steps, train_loss, alpha=0.5, label="Train", color="steelblue")
    if any(v is not None for v in val_loss):
        vals = [(s, v) for s, v in zip(steps, val_loss) if v is not None]
        if vals:
            ax.plot(*zip(*vals), "o-", label="Val", color="darkorange", markersize=4)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training Progress")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Learning rate
    ax = fig.add_subplot(gs[0, 1])
    if any(v is not None for v in lr):
        ax.plot(steps, lr, color="green")
    ax.set_xlabel("Step")
    ax.set_ylabel("Learning Rate")
    ax.set_title("LR Schedule")
    ax.grid(True, alpha=0.3)

    # M effective rank
    ax = fig.add_subplot(gs[1, 0])
    if any(v is not None for v in m_rank):
        vals = [(s, v) for s, v in zip(steps, m_rank) if v is not None]
        if vals:
            ax.plot(*zip(*vals), "o-", color="steelblue", markersize=4)
    ax.set_xlabel("Step")
    ax.set_ylabel("Effective Rank (90%)")
    ax.set_title("M Crystallisation")
    ax.grid(True, alpha=0.3)

    # M condition number
    ax = fig.add_subplot(gs[1, 1])
    if any(v is not None for v in m_cond):
        vals = [(s, v) for s, v in zip(steps, m_cond) if v is not None]
        if vals:
            ax.semilogy(*zip(*vals), "o-", color="darkorange", markersize=4)
    ax.axhline(2, color="red", ls="--", alpha=0.5, label="Kill: κ < 2")
    ax.set_xlabel("Step")
    ax.set_ylabel("Condition Number")
    ax.set_title("M Spectral Spread")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"{model_name} Training Report", fontsize=14, fontweight="bold")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
