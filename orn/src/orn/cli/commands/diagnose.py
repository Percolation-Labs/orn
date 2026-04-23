"""orn diagnose — spectral analysis of the shared coupling M on a checkpoint."""
from __future__ import annotations

from pathlib import Path


def cmd(args) -> None:
    import numpy as np
    from orn.utils import load_checkpoint, resolve_local
    from orn.diagnostics.spectral import spectral_analysis

    ckpt_path, _ = resolve_local(args.checkpoint)
    ckpt = load_checkpoint(ckpt_path, map_location="cpu")
    state = ckpt.get("model", ckpt)  # legacy checkpoints may be raw state dicts
    if "A" not in state or "B" not in state:
        raise ValueError("Checkpoint does not carry SharedM (A, B). `orn diagnose` is ORN-only.")

    A = state["A"].double().numpy()
    B = state["B"].double().numpy()
    with np.errstate(all="ignore"):
        M = A @ B.T
    spectral_analysis(M, verbose=True)

    if args.plot:
        from orn.diagnostics.plots import plot_spectrum
        plot_spectrum(M, title=f"M Spectrum ({args.checkpoint})",
                       save_path=Path(args.plot))
        print(f"Plot: {args.plot}")


def register(subparsers) -> None:
    sp = subparsers.add_parser("diagnose", help="Spectral diagnostics on M")
    sp.add_argument("--checkpoint", required=True,
                    help="Local path or known name (orn-v2-108m, orn-v3-605m, ...)")
    sp.add_argument("--plot", default=None, metavar="PATH",
                    help="Save a spectrum plot to the given path")
    sp.set_defaults(fn=cmd)
