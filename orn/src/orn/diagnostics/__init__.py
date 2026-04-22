"""
Diagnostics suite for ORN models.

- spectral.py: M's spectral analysis (SVD, eigenvalues, effective rank, condition number)
- coupling.py: Cross-layer coupling invariance analysis
- plots.py: Visualization utilities for spectral evolution, attention, scaling curves
"""

from __future__ import annotations
from orn.diagnostics.spectral import spectral_analysis, spectral_trajectory
from orn.diagnostics.coupling import coupling_invariance, extract_coupling_matrices
from orn.diagnostics.plots import (
    plot_spectrum,
    plot_spectral_trajectory,
    plot_attention,
    plot_scaling_curves,
    plot_training_report,
)
