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
from orn.diagnostics.style import apply_style, PALETTE, CMAP_SEQ

# Apply shared style on import so every figure matches.
apply_style()


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


def plot_layer_invariance(
    per_layer_M: list[np.ndarray],
    titles: list[str] | None = None,
    save_path: str | Path | None = None,
    suptitle: str = "Per-layer coupling eigenspectra — the motivation for shared M",
):
    """The motivating figure for shared M.

    Left : overlay of sorted eigenvalue magnitudes for each layer's
           M_l = W_Q^T W_K. Curves should lie on top of each other — the
           *rule* is invariant — even when the raw matrices differ.
    Right: Spearman-correlation heatmap of those eigenvalue curves across
           layers. Values near 1 = identical structure; near 0 = unrelated.

    Works for any sequence of square matrices. Use with a trained transformer
    (one model, many layers) or across model families (concatenate a layer
    from each).
    """
    from scipy.stats import spearmanr

    L = len(per_layer_M)
    if titles is None:
        titles = [f"layer {i}" for i in range(L)]

    with np.errstate(all="ignore"):
        eig_mags = [np.sort(np.abs(np.linalg.eigvals(M.astype(np.float64))))[::-1]
                    for M in per_layer_M]
        # Pad/truncate to common length for heatmap alignment
        d = min(len(e) for e in eig_mags)
        eig_mags_trim = [e[:d] for e in eig_mags]
        corr = np.zeros((L, L))
        for i in range(L):
            for j in range(L):
                corr[i, j], _ = spearmanr(eig_mags_trim[i], eig_mags_trim[j])

    apply_style()
    fig = plt.figure(figsize=(12, 4.8))
    gs = GridSpec(1, 2, width_ratios=[1.4, 1], wspace=0.28)

    ax0 = fig.add_subplot(gs[0])
    depth_cmap = plt.get_cmap("viridis", max(L, 2))
    for i, (e, title) in enumerate(zip(eig_mags, titles)):
        ax0.semilogy(e / (e[0] + 1e-12), color=depth_cmap(i),
                     alpha=0.85, linewidth=1.4, label=title)
    ax0.set_xlabel("rank (sorted)")
    ax0.set_ylabel("|eigenvalue| (normalised)")
    ax0.set_title("Sorted eigenvalue magnitudes per layer")
    if L <= 12:
        ax0.legend(fontsize=8, loc="upper right", ncol=2)

    ax1 = fig.add_subplot(gs[1])
    im = ax1.imshow(corr, vmin=0.0, vmax=1.0, cmap=CMAP_SEQ, aspect="auto")
    ax1.set_xticks(range(L)); ax1.set_yticks(range(L))
    ax1.set_xticklabels(titles, rotation=90, fontsize=8)
    ax1.set_yticklabels(titles, fontsize=8)
    ax1.set_title(f"Spearman ρ across layers (mean off-diag {_mean_offdiag(corr):.3f})")
    ax1.grid(False)
    plt.colorbar(im, ax=ax1, shrink=0.85, label="ρ")

    fig.suptitle(suptitle, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        fig.savefig(save_path)
    return fig


def _mean_offdiag(C: np.ndarray) -> float:
    n = C.shape[0]
    if n < 2:
        return float("nan")
    mask = ~np.eye(n, dtype=bool)
    return float(C[mask].mean())


def plot_memoise_vs_store(
    memoise_history: list[float],
    store_history: list[float],
    mem_params: int,
    sto_params: int,
    save_path: str | Path | None = None,
):
    """Colour-matching / memoise-vs-store comparison: loss curves + param-budget bar."""
    apply_style()
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 4),
                                    gridspec_kw={"width_ratios": [2, 1]})

    # Lightly smooth the loss curves so the eye reads trend, not step noise.
    def _smooth(xs, w=9):
        if len(xs) < w:
            return xs
        import numpy as _np
        c = _np.convolve(_np.asarray(xs), _np.ones(w)/w, mode="valid")
        pad = (len(xs) - len(c)) // 2
        return [float(x) for x in _np.concatenate([xs[:pad], c, xs[len(xs)-pad:]])][:len(xs)]

    ax0.plot(_smooth(memoise_history), label="MEMOISE (shared AB^T)",
             color=PALETTE[0], linewidth=2.0)
    ax0.plot(_smooth(store_history), label="STORE (per-layer M_l)",
             color=PALETTE[1], linewidth=2.0, alpha=0.9)
    ax0.set_xlabel("training step"); ax0.set_ylabel("loss")
    ax0.set_title("Training loss")
    ax0.legend(loc="upper right")

    bars = ax1.bar(["MEMOISE", "STORE"], [mem_params, sto_params],
                    color=[PALETTE[0], PALETTE[1]], width=0.6)
    ax1.set_ylabel("coupling parameters")
    ax1.set_title(f"Coupling budget ({sto_params/max(mem_params,1):.1f}x compression)")
    for b, v in zip(bars, [mem_params, sto_params]):
        ax1.text(b.get_x() + b.get_width()/2, v, f"{v:,}",
                 ha="center", va="bottom", fontsize=9)
    ax1.margins(y=0.18)

    fig.suptitle("Colour matching: MEMOISE matches STORE on a provably-invariant task",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_rank_trajectory(trajectory: list[dict], d_model: int,
                          save_path: str | Path | None = None):
    """SPE-02: eff_rank_90 of M across training steps."""
    steps = [t["step"] for t in trajectory]
    ranks = [t["eff_rank_90"] for t in trajectory]
    conds = [t["condition_number"] for t in trajectory]

    apply_style()
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 4))

    ax0.plot(steps, ranks, marker="o", linewidth=2.0, color=PALETTE[0])
    ax0.axhline(d_model, color="#888", linestyle=":", label=f"d_model = {d_model}")
    ax0.axhline(d_model / 2, color="#888", linestyle="--", alpha=0.6, label="d/2")
    ax0.set_xlabel("training step"); ax0.set_ylabel("effective rank (90% energy)")
    ax0.set_title("M crystallisation trajectory")
    ax0.legend()

    ax1.semilogy(steps, conds, marker="o", linewidth=2.0, color=PALETTE[1])
    ax1.set_xlabel("training step"); ax1.set_ylabel("condition number σ_max/σ_min")
    ax1.set_title("M becomes ill-conditioned (spectral structure emerges)")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig
