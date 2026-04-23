"""orn eval / bench / bench-summary — lm-eval-harness wrappers."""
from __future__ import annotations

import json
import time
from pathlib import Path


def _load_any_checkpoint(checkpoint: str, device):
    """Load either a known-name or a local-path checkpoint and return a ready model."""
    from orn.utils import load_checkpoint, resolve_local, load_known
    from orn.models import build_model

    ckpt_path, spec = resolve_local(checkpoint)
    if spec is not None:
        model, spec = load_known(spec.name, device=device)
        return model, spec
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    model_cfg = ckpt.get("model_config") or ckpt.get("config", {}).get("model", {})
    arch = dict(model_cfg).pop("arch", "orn_v2")
    model = build_model(arch, dict(model_cfg))
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, None


def _summarise(results: dict) -> dict:
    out = {}
    for task, scores in results.items():
        out[task] = {k.split(",")[0]: v for k, v in scores.items()
                      if isinstance(v, (int, float)) and not k.endswith("stderr")}
    return out


# ─────────────────────────── eval ────────────────────────────

def cmd_eval(args) -> None:
    from orn.utils import pick_device
    from orn.utils.hf import get_tokenizer
    from orn.eval import eval_benchmarks

    device = pick_device(args.device)
    model, _ = _load_any_checkpoint(args.checkpoint, device)
    tok = get_tokenizer(args.tokenizer)
    tasks = args.tasks.split(",") if args.tasks else None
    res = eval_benchmarks(model, tok, tasks=tasks, limit=args.limit,
                           max_length=args.seq, device=device)
    print(json.dumps(res, indent=2, default=str))


# ─────────────────────────── bench ───────────────────────────

def cmd_bench(args) -> None:
    from orn.utils import load_known, pick_device
    from orn.utils.hf import get_tokenizer
    from orn.eval import eval_benchmarks, DEFAULT_TASKS

    device = pick_device(args.device)
    print(f"device: {device}")
    print(f"loading: {args.checkpoint}")
    t0 = time.time()
    model, spec = load_known(args.checkpoint, device=device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"loaded {spec.name} in {time.time() - t0:.1f}s "
          f"({n_params:,} params, arch={spec.arch})")

    tok = get_tokenizer(args.tokenizer)
    tasks = [t.strip() for t in args.tasks.split(",")] if args.tasks else DEFAULT_TASKS
    print(f"tasks: {tasks}   limit: {args.limit}   seq: {args.seq}")

    tb = time.time()
    results = eval_benchmarks(model, tok, tasks=tasks, limit=args.limit,
                               max_length=args.seq, device=device)
    dt = time.time() - tb

    summary = _summarise(results)
    record = {
        "checkpoint": spec.name, "arch": spec.arch, "n_params": n_params,
        "training_tokens": spec.training_tokens,
        "limit": args.limit, "seq": args.seq, "device": str(device),
        "runtime_sec": round(dt, 1), "tasks": summary,
    }
    out_dir = Path("bench/results"); out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    fname = Path(args.out) if args.out else out_dir / f"{spec.name}_{stamp}.json"
    fname.write_text(json.dumps(record, indent=2))

    print("\n=== results ===")
    for task, row in summary.items():
        items = "  ".join(f"{k}={v:.4f}" for k, v in row.items() if isinstance(v, float))
        print(f"  {task:<40s} {items}")
    print(f"\nsaved: {fname}  ({dt / 60:.1f} min)")


# ─────────────────────────── bench-summary ───────────────────

def cmd_bench_summary(args) -> None:
    from orn.eval.summarise import load_runs, format_text, format_markdown, format_latex
    runs = load_runs(args.dir)
    if not runs:
        print(f"no benchmark JSON files found in {args.dir}"); return
    if args.format in ("text", "all"):
        print("\n=== plain text ===\n" + format_text(runs))
    if args.format in ("md", "all"):
        print("\n=== markdown ===\n" + format_markdown(runs))
    if args.format in ("latex", "all"):
        print("\n=== latex ===\n" + format_latex(runs))


def register(subparsers) -> None:
    # eval
    sp = subparsers.add_parser("eval", help="lm-eval benchmarks on a single checkpoint")
    sp.add_argument("--checkpoint", required=True,
                    help="Local path or known name (orn-v2-108m, orn-v3-605m, ...)")
    sp.add_argument("--tasks", default=None,
                    help="Comma-separated (default: DEFAULT_TASKS)")
    sp.add_argument("--limit", type=int, default=None, help="per-task cap")
    sp.add_argument("--seq", type=int, default=None, help="max sequence length")
    sp.add_argument("--device", default=None)
    sp.add_argument("--tokenizer", default="tiktoken-gpt2")
    sp.set_defaults(fn=cmd_eval)

    # bench
    sp = subparsers.add_parser("bench",
                                help="Full benchmark run on a known checkpoint, saves JSON")
    sp.add_argument("checkpoint", help="known name, e.g. orn-v3-605m")
    sp.add_argument("--tasks", default=None,
                    help="Comma-separated (default: DEFAULT_TASKS)")
    sp.add_argument("--limit", type=int, default=None)
    sp.add_argument("--seq", type=int, default=None)
    sp.add_argument("--device", default=None)
    sp.add_argument("--tokenizer", default="tiktoken-gpt2")
    sp.add_argument("--out", default=None, help="output JSON path")
    sp.set_defaults(fn=cmd_bench)

    # bench-summary
    sp = subparsers.add_parser("bench-summary",
                                help="Collate bench/results/*.json into paper-ready tables")
    sp.add_argument("--dir", default="bench/results")
    sp.add_argument("--format", choices=["text", "md", "latex", "all"], default="all")
    sp.set_defaults(fn=cmd_bench_summary)
