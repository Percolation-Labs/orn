"""Optional Plotly backend for the motivating figures.

Enabled when `pip install -e '.[plots]'` brings in plotly (and kaleido for
static PNG export). Mirrors the matplotlib helpers in `plots.py` for the
three figures that benefit most from interactivity (coupling-manifold
overlay, memoise-vs-store, rank trajectory).

Selection is via the CLI flag `orn reproduce ... --backend plotly`. Output
writes both an interactive `.html` file and a static `.png` next to it.
"""
from __future__ import annotations

from pathlib import Path


# Seaborn-muted palette, matched to orn.diagnostics.style.PALETTE for
# consistency across matplotlib and Plotly outputs.
PALETTE = [
    "#4C72B0", "#DD8452", "#55A467", "#C44E52",
    "#8172B2", "#937860", "#DA8BC3", "#8C8C8C",
    "#CCB974", "#64B5CD",
]


def _require_plotly():
    try:
        import plotly.graph_objects as go  # noqa: F401
        import plotly.io as pio            # noqa: F401
    except ImportError as e:
        raise ImportError(
            "Plotly backend requires the `[plots]` extras: "
            "`pip install -e '.[plots]'`"
        ) from e


def _apply_layout(fig, title: str, height: int = 480):
    fig.update_layout(
        title=dict(text=title, x=0.02, xanchor="left",
                   font=dict(size=14, family="Helvetica Neue, Arial")),
        template="plotly_white",
        margin=dict(l=60, r=30, t=60, b=50),
        height=height,
        font=dict(family="Helvetica Neue, Arial", size=12, color="#333"),
        legend=dict(bgcolor="rgba(255,255,255,0)"),
    )
    fig.update_xaxes(showline=True, linecolor="#555", ticks="outside",
                      gridcolor="#e9e9e9", zeroline=False)
    fig.update_yaxes(showline=True, linecolor="#555", ticks="outside",
                      gridcolor="#e9e9e9", zeroline=False)
    return fig


def _write(fig, save_path: str | Path | None):
    if save_path is None:
        return fig
    p = Path(save_path)
    # Always write HTML. If requested path has .png, also render static via kaleido.
    html_path = p.with_suffix(".html")
    fig.write_html(html_path, include_plotlyjs="cdn", full_html=False)
    if p.suffix == ".png":
        try:
            fig.write_image(p)
        except Exception:
            # kaleido not installed, skip silently — HTML still written.
            pass
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# Figures
# ═══════════════════════════════════════════════════════════════════════════════

def plot_layer_invariance(per_layer_M, titles=None, save_path=None,
                           suptitle="Per-layer coupling eigenspectra"):
    """Plotly version of the motivating-figure overlay + heatmap."""
    _require_plotly()
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from scipy.stats import spearmanr

    L = len(per_layer_M)
    if titles is None:
        titles = [f"layer {i}" for i in range(L)]

    with np.errstate(all="ignore"):
        eig_mags = [np.sort(np.abs(np.linalg.eigvals(M.astype(np.float64))))[::-1]
                    for M in per_layer_M]
        d = min(len(e) for e in eig_mags)
        trim = [e[:d] for e in eig_mags]
        corr = np.zeros((L, L))
        for i in range(L):
            for j in range(L):
                corr[i, j], _ = spearmanr(trim[i], trim[j])

    fig = make_subplots(rows=1, cols=2, column_widths=[0.6, 0.4],
                        subplot_titles=(
                            "Sorted |eigenvalue| per layer (log y)",
                            f"Spearman ρ across layers (mean off-diag "
                            f"{_mean_offdiag(corr):.3f})",
                        ))

    import plotly.express as px
    colorscale = px.colors.sequential.Viridis
    for i, (e, title) in enumerate(zip(eig_mags, titles)):
        colour = colorscale[int(i / max(L-1, 1) * (len(colorscale)-1))]
        fig.add_trace(
            go.Scatter(
                x=list(range(len(e))),
                y=(e / (e[0] + 1e-12)).tolist(),
                mode="lines", name=title,
                line=dict(color=colour, width=1.6),
                hovertemplate=f"{title}<br>rank=%{{x}}<br>|λ|=%{{y:.4f}}<extra></extra>",
            ),
            row=1, col=1,
        )

    fig.add_trace(
        go.Heatmap(
            z=corr, x=titles, y=titles,
            colorscale="Viridis", zmin=0.0, zmax=1.0,
            colorbar=dict(title="ρ", thickness=12),
            hovertemplate="ρ(%{x}, %{y}) = %{z:.3f}<extra></extra>",
        ),
        row=1, col=2,
    )

    fig.update_yaxes(type="log", row=1, col=1, title="|eigenvalue|")
    fig.update_xaxes(title="rank", row=1, col=1)
    _apply_layout(fig, suptitle, height=440)
    return _write(fig, save_path)


def plot_memoise_vs_store(memoise_history, store_history,
                           mem_params: int, sto_params: int, save_path=None):
    """Plotly version of MEMOISE vs STORE."""
    _require_plotly()
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=1, cols=2, column_widths=[0.65, 0.35],
                        subplot_titles=(
                            "Training loss",
                            f"Coupling budget ({sto_params/max(mem_params,1):.1f}x compression)",
                        ))

    fig.add_trace(go.Scatter(y=memoise_history, mode="lines",
                             name="MEMOISE (shared AB^T)",
                             line=dict(color=PALETTE[0], width=2.4)),
                  row=1, col=1)
    fig.add_trace(go.Scatter(y=store_history, mode="lines",
                             name="STORE (per-layer M_l)",
                             line=dict(color=PALETTE[1], width=2.4)),
                  row=1, col=1)

    fig.add_trace(go.Bar(
        x=["MEMOISE", "STORE"], y=[mem_params, sto_params],
        marker_color=[PALETTE[0], PALETTE[1]],
        text=[f"{mem_params:,}", f"{sto_params:,}"],
        textposition="outside",
        showlegend=False,
    ), row=1, col=2)

    fig.update_xaxes(title="training step", row=1, col=1)
    fig.update_yaxes(title="loss", row=1, col=1)
    fig.update_yaxes(title="coupling parameters", row=1, col=2)

    _apply_layout(fig,
                  "Colour matching: MEMOISE matches STORE on a provably-invariant task",
                  height=440)
    return _write(fig, save_path)


def plot_rank_trajectory(trajectory, d_model: int, save_path=None):
    """Plotly version of SPE-02."""
    _require_plotly()
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    steps = [t["step"] for t in trajectory]
    ranks = [t["eff_rank_90"] for t in trajectory]
    conds = [t["condition_number"] for t in trajectory]

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("M crystallisation trajectory",
                                        "Condition number (log y)"))

    fig.add_trace(go.Scatter(x=steps, y=ranks, mode="lines+markers",
                             line=dict(color=PALETTE[0], width=2.4),
                             marker=dict(size=7), name="eff rank (90%)"),
                  row=1, col=1)
    fig.add_hline(y=d_model, line=dict(color="#888", dash="dot"),
                  annotation_text=f"d_model = {d_model}",
                  annotation_position="top right", row=1, col=1)
    fig.add_hline(y=d_model/2, line=dict(color="#888", dash="dash"),
                  annotation_text="d/2", annotation_position="top right",
                  row=1, col=1)

    fig.add_trace(go.Scatter(x=steps, y=conds, mode="lines+markers",
                             line=dict(color=PALETTE[1], width=2.4),
                             marker=dict(size=7), name="σ_max/σ_min",
                             showlegend=False),
                  row=1, col=2)

    fig.update_xaxes(title="training step")
    fig.update_yaxes(title="effective rank (90% energy)", row=1, col=1)
    fig.update_yaxes(title="condition number", type="log", row=1, col=2)

    _apply_layout(fig, "M crystallisation and conditioning", height=440)
    return _write(fig, save_path)


def _mean_offdiag(C):
    import numpy as np
    n = C.shape[0]
    if n < 2:
        return float("nan")
    mask = ~np.eye(n, dtype=bool)
    return float(C[mask].mean())
