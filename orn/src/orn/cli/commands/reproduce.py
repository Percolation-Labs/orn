"""orn reproduce — run registered key results and optionally save plots."""
from __future__ import annotations

from pathlib import Path


def cmd(args) -> None:
    from orn.reproduce import list_results, resolve

    if args.list or not args.names:
        tier = "slow_reproduce" if args.slow else None
        results = list_results(tier=tier) if args.slow else list_results()
        print(f"{'CODE':<8s} {'TIER':<16s} {'RUNTIME':<8s} {'PLOT':<5s} SLUG")
        for r in results:
            plot = "✓" if r.has_plot else "."
            print(f"{r.code:<8s} {r.tier:<16s} {r.runtime_s:<5d}s   {plot:<5s} {r.slug}")
        print("\nUsage: orn reproduce <code|slug|prefix> [...] [--save-plots DIR]")
        return

    save_dir = Path(args.save_plots) if args.save_plots else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    for r in resolve(args.names):
        print(f"\n── {r.code}  {r.slug} ──")
        out = r.run(device=args.device)
        for k, v in out.items():
            if k.startswith("_"):  # hide internal payloads (_per_layer_M, etc.)
                continue
            s = f"{v:.4f}" if isinstance(v, float) else str(v)
            print(f"    {k}: {s}")
        if save_dir and r.has_plot:
            ext = ".png" if args.backend == "matplotlib" else ".html"
            path = save_dir / f"{r.code}_{r.slug}{ext}"
            r.plot(out, save_path=path, backend=args.backend)
            print(f"    plot: {path}")


def register(subparsers) -> None:
    sp = subparsers.add_parser("reproduce", help="Run reproducible key results")
    sp.add_argument("names", nargs="*")
    sp.add_argument("--list", action="store_true")
    sp.add_argument("--slow", action="store_true", help="Include slow-tier results")
    sp.add_argument("--save-plots", default=None, metavar="DIR",
                    help="Save plots for every result that defines plot()")
    sp.add_argument("--backend", choices=["matplotlib", "plotly"], default="matplotlib",
                    help="Plot backend. `plotly` writes interactive HTML "
                         "(requires pip install -e '.[plots]').")
    sp.add_argument("--device", default=None)
    sp.set_defaults(fn=cmd)
