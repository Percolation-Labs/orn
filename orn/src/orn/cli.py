"""
ORN Command-Line Interface.

Subcommands:
  orn configs                   list available training presets
  orn prepare <dataset>         download + tokenise training data
  orn train --config NAME       train a model (preset or path to JSON)
  orn predict --checkpoint P    generate text from a checkpoint
  orn diagnose --checkpoint P   spectral diagnostics on M
  orn eval --checkpoint P       lm-eval benchmarks (hellaswag/piqa/arc)
  orn reproduce [--list] [...]  run reproducible key results
  orn push --checkpoint P       upload checkpoint to HuggingFace
  orn pull --model NAME         pull baseline model from HuggingFace
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════════════════
# configs
# ═══════════════════════════════════════════════════════════════════════════════

def cmd_configs(args):
    from orn.training import list_configs, load_config, CONFIG_DIR
    print(f"Training configs ({CONFIG_DIR}):")
    for name in list_configs():
        cfg = load_config(name)
        desc = cfg.get("_description", "")
        arch = cfg.get("model", {}).get("arch", "?")
        print(f"  {name:<30s}  [{arch}]  {desc}")


# ═══════════════════════════════════════════════════════════════════════════════
# prepare
# ═══════════════════════════════════════════════════════════════════════════════

def cmd_prepare(args):
    from orn.data.prepare import prepare_tinystories, prepare_openwebtext
    if args.dataset == "tinystories":
        prepare_tinystories(output_dir=args.output, max_tokens=args.max_tokens)
    elif args.dataset in ("openwebtext", "fineweb-edu"):
        prepare_openwebtext(output_dir=args.output, max_tokens=args.max_tokens)
    else:
        print(f"Unknown dataset '{args.dataset}'. Options: tinystories, openwebtext, fineweb-edu")
        sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════════
# train
# ═══════════════════════════════════════════════════════════════════════════════

def _ensure_data_dir(data_dir: str, config_name: str, seq: int,
                      batch_size: int, vocab_size: int) -> None:
    """If using dry_run and no shards exist, drop a small synthetic shard.

    Real configs (orn_v*, lorn_v4_*) should use `orn prepare` instead; we do
    not silently synthesise data for them.
    """
    import torch
    from pathlib import Path
    p = Path(data_dir)
    if p.exists() and any(p.glob("shard_*.pt")):
        return
    if config_name != "dry_run":
        return
    p.mkdir(parents=True, exist_ok=True)
    n_tokens = max(batch_size * seq * 400, 50_000)
    torch.save(torch.randint(0, vocab_size, (n_tokens,), dtype=torch.int32),
               p / "shard_0000.pt")
    print(f"  [dry_run] synthesised {n_tokens:,} random tokens at {p / 'shard_0000.pt'}")


def _apply_overrides(model_kw: dict, training_kw: dict, args):
    for k in ("d_model", "n_layers", "n_heads", "n_q_heads", "n_kv_heads",
             "d_head", "seq_len", "d_corr", "vocab_size", "ffn_width_mult"):
        v = getattr(args, k.replace("-", "_"), None)
        if v is not None:
            model_kw[k] = v
    for k in ("lr", "batch_size", "grad_accum_steps", "max_tokens", "warmup_steps"):
        v = getattr(args, k.replace("-", "_"), None)
        if v is not None:
            training_kw[k] = v
    if args.no_compile:        training_kw["compile"] = False
    if args.no_mixed_precision: training_kw["mixed_precision"] = False
    if args.output:             training_kw["output_dir"] = args.output


def cmd_train(args):
    from orn.training import load_config, Trainer, TrainingConfig, get_loss_fn
    from orn.models import build_model
    from orn.data.loaders import ShardedDataLoader
    from orn.utils import pick_device, load_env
    load_env()

    cfg = load_config(args.config)
    model_kw = dict(cfg.get("model", {}))
    training_kw = dict(cfg.get("training", {}))
    _apply_overrides(model_kw, training_kw, args)

    arch = model_kw.pop("arch", "orn_v2")
    model = build_model(arch, model_kw)

    train_cfg = TrainingConfig.from_dict(training_kw)
    device = pick_device(args.device)

    data_dir = args.data or cfg.get("data", {}).get("path") or "data"
    seq = model_kw.get("seq_len") or model_kw.get("T_max") or train_cfg.seq_len

    # Zero-friction dry run: if no shards exist and this is the dry_run config,
    # synthesize one so new users can smoke the pipeline with no HF token.
    _ensure_data_dir(data_dir, args.config, seq, train_cfg.batch_size,
                     vocab_size=model_kw.get("vocab_size", 50257))

    train_loader = ShardedDataLoader(data_dir, seq, train_cfg.batch_size, device)
    val_loader = ShardedDataLoader(data_dir, seq, train_cfg.batch_size, device, eval_mode=True)

    loss_fn = get_loss_fn(cfg.get("loss", {}).get("type", "lm_ce"))
    trainer = Trainer(model, train_cfg, train_loader, val_loader,
                      device=device, resume_from=args.resume, loss_fn=loss_fn)
    trainer.train()


# ═══════════════════════════════════════════════════════════════════════════════
# predict / diagnose
# ═══════════════════════════════════════════════════════════════════════════════

def cmd_predict(args):
    import torch
    from orn.utils import load_checkpoint, pick_device, resolve_local, load_known
    from orn.utils.hf import get_tokenizer
    from orn.models import build_model

    device = pick_device(args.device)

    # Accept either a local path or a known-checkpoint name.
    ckpt_path, spec = resolve_local(args.checkpoint)
    if spec is not None:
        # Known checkpoint: use registered arch + config, bypass model_config parsing.
        model, _ = load_known(spec.name, device=device)
    else:
        ckpt = load_checkpoint(ckpt_path, map_location=device)
        model_cfg = ckpt.get("model_config") or ckpt.get("config", {}).get("model", {})
        if not isinstance(model_cfg, dict):
            raise ValueError("Unsupported checkpoint: model_config is not a dict")
        arch = model_cfg.pop("arch", None)
        if arch is None:
            raise ValueError("Checkpoint has no model_config.arch — re-train with the new trainer "
                             "or use a known checkpoint name (orn pull-checkpoint --list)")
        model = build_model(arch, model_cfg)
        model.load_state_dict(ckpt["model"])
        model = model.to(device).eval()

    enc = get_tokenizer(args.tokenizer)
    prompt = args.prompt or "The"
    ids = torch.tensor(enc.encode(prompt) if hasattr(enc, "encode") else enc(prompt)["input_ids"],
                       device=device)
    out = model.generate(ids, max_new=args.max_tokens,
                         temperature=args.temperature, top_k=args.top_k)
    text = enc.decode(out.tolist())
    print(text)


def cmd_diagnose(args):
    import numpy as np
    import torch
    from orn.utils import load_checkpoint, resolve_local
    from orn.diagnostics.spectral import spectral_analysis

    ckpt_path, _ = resolve_local(args.checkpoint)
    ckpt = load_checkpoint(ckpt_path, map_location="cpu")
    state = ckpt.get("model", ckpt)   # legacy checkpoints may be raw state dicts
    if "A" not in state or "B" not in state:
        raise ValueError("Checkpoint doesn't carry SharedM (A, B). `orn diagnose` is ORN-only.")
    A = state["A"].double().numpy()
    B = state["B"].double().numpy()
    with np.errstate(all="ignore"):
        M = A @ B.T
    spectral_analysis(M, verbose=True)

    if args.plot:
        from orn.diagnostics.plots import plot_spectrum
        plot_spectrum(M, title=f"M Spectrum ({args.checkpoint})", save_path=Path(args.plot))
        print(f"Plot: {args.plot}")


# ═══════════════════════════════════════════════════════════════════════════════
# eval / push / pull
# ═══════════════════════════════════════════════════════════════════════════════

def cmd_eval(args):
    import torch
    from orn.utils import load_checkpoint, pick_device
    from orn.utils.hf import get_tokenizer
    from orn.models import build_model
    from orn.eval import eval_benchmarks

    device = pick_device(args.device)
    ckpt = load_checkpoint(args.checkpoint, map_location=device)
    model_cfg = ckpt.get("model_config") or ckpt.get("config", {}).get("model", {})
    arch = dict(model_cfg).pop("arch", "orn_v2")
    model = build_model(arch, dict(model_cfg))
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    tok = get_tokenizer(args.tokenizer)
    tasks = args.tasks.split(",") if args.tasks else None
    res = eval_benchmarks(model, tok, tasks=tasks)
    print(json.dumps(res, indent=2, default=str))


def cmd_push(args):
    from orn.utils.hf import push_checkpoint
    url = push_checkpoint(args.checkpoint, args.repo, private=not args.public)
    print(f"Uploaded: {url}")


def cmd_pull(args):
    from orn.utils.hf import pull_model
    mdl, tok = pull_model(args.model, cache_dir=args.cache)
    print(f"Pulled {args.model}: {type(mdl).__name__}, vocab={tok.vocab_size}")


def cmd_pull_checkpoint(args):
    from orn.utils import known_checkpoints, fetch_checkpoint
    if args.list or not args.name:
        print(f"{'NAME':<28s} {'ARCH':<10s} DESCRIPTION")
        for s in known_checkpoints():
            print(f"  {s.name:<26s} {s.arch:<10s} {s.description}")
        print("\nDefault: use the unsuffixed name (e.g. `orn-v3-605m`) — that's the "
              "latest/most-trained\ncheckpoint for each model. Suffixed names "
              "(-1B, -2B, -branch-2B, ...) pin to a specific\ntraining marker.")
        print("\nUsage: orn pull-checkpoint <name>")
        return
    path = fetch_checkpoint(args.name, cache_dir=args.cache)
    print(f"Cached at: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# reproduce
# ═══════════════════════════════════════════════════════════════════════════════

def cmd_reproduce(args):
    from orn.reproduce import list_results, resolve
    if args.list or not args.names:
        tier = "slow_reproduce" if args.slow else None
        results = list_results(tier=tier) if args.slow else list_results()
        print(f"{'CODE':<8s} {'TIER':<16s} {'RUNTIME':<8s} {'PLOT':<5s} SLUG")
        for r in results:
            plot = "✓" if r.has_plot else "."
            print(f"{r.code:<8s} {r.tier:<16s} {r.runtime_s:<5d}s   "
                  f"{plot:<5s} {r.slug}")
        print("\nUsage: orn reproduce <code|slug|prefix> [...] [--save-plots DIR]")
        return

    save_dir = Path(args.save_plots) if args.save_plots else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    results = resolve(args.names)
    for r in results:
        print(f"\n── {r.code}  {r.slug} ──")
        out = r.run(device=args.device)
        for k, v in out.items():
            if k.startswith("_"):   # internal payloads (e.g. _per_layer_M)
                continue
            s = f"{v:.4f}" if isinstance(v, float) else str(v)
            print(f"    {k}: {s}")
        if save_dir and r.has_plot:
            path = save_dir / f"{r.code}_{r.slug}.png"
            r.plot(out, save_path=path)
            print(f"    plot: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# parser
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(prog="orn", description="Orbital Response Network CLI")
    sub = p.add_subparsers(dest="command")

    # ── configs ──
    sp = sub.add_parser("configs", help="List training presets")
    sp.set_defaults(fn=cmd_configs)

    # ── prepare ──
    sp = sub.add_parser("prepare", help="Download + tokenise data")
    sp.add_argument("dataset", choices=["tinystories", "openwebtext", "fineweb-edu"])
    sp.add_argument("--output", default="data")
    sp.add_argument("--max-tokens", type=int, default=50_000_000)
    sp.set_defaults(fn=cmd_prepare)

    # ── train ──
    sp = sub.add_parser("train", help="Train a model")
    sp.add_argument("--config", required=True, help="Preset name or path to JSON")
    sp.add_argument("--resume", default=None)
    sp.add_argument("--device", default=None)
    sp.add_argument("--data", default=None)
    sp.add_argument("--output", default=None)
    for k in ("d_model", "n_layers", "n_heads", "n_q_heads", "n_kv_heads",
             "d_head", "seq_len", "d_corr", "vocab_size", "ffn_width_mult"):
        sp.add_argument(f"--{k.replace('_', '-')}", type=int, default=None)
    sp.add_argument("--lr", type=float, default=None)
    sp.add_argument("--batch-size", type=int, default=None)
    sp.add_argument("--grad-accum-steps", type=int, default=None)
    sp.add_argument("--max-tokens", type=int, default=None)
    sp.add_argument("--warmup-steps", type=int, default=None)
    sp.add_argument("--no-compile", action="store_true")
    sp.add_argument("--no-mixed-precision", action="store_true")
    sp.set_defaults(fn=cmd_train)

    # ── predict ──
    sp = sub.add_parser("predict", help="Generate text from a checkpoint")
    sp.add_argument("--checkpoint", required=True)
    sp.add_argument("--prompt", default="The")
    sp.add_argument("--max-tokens", type=int, default=200)
    sp.add_argument("--temperature", type=float, default=0.8)
    sp.add_argument("--top-k", type=int, default=40)
    sp.add_argument("--device", default=None)
    sp.add_argument("--tokenizer", default="tiktoken-gpt2")
    sp.set_defaults(fn=cmd_predict)

    # ── diagnose ──
    sp = sub.add_parser("diagnose", help="Spectral diagnostics on M")
    sp.add_argument("--checkpoint", required=True)
    sp.add_argument("--plot", default=None)
    sp.set_defaults(fn=cmd_diagnose)

    # ── eval ──
    sp = sub.add_parser("eval", help="Benchmarks (hellaswag, piqa, arc_easy)")
    sp.add_argument("--checkpoint", required=True)
    sp.add_argument("--tasks", default=None,
                    help="Comma-separated list (default: hellaswag,piqa,arc_easy)")
    sp.add_argument("--device", default=None)
    sp.add_argument("--tokenizer", default="tiktoken-gpt2")
    sp.set_defaults(fn=cmd_eval)

    # ── push / pull ──
    sp = sub.add_parser("push", help="Upload checkpoint to HuggingFace")
    sp.add_argument("--checkpoint", required=True)
    sp.add_argument("--repo", required=True)
    sp.add_argument("--public", action="store_true")
    sp.set_defaults(fn=cmd_push)

    sp = sub.add_parser("pull", help="Pull baseline model from HuggingFace")
    sp.add_argument("--model", required=True,
                    help="gpt2, smollm2-135m, pythia-70m, pythia-410m")
    sp.add_argument("--cache", default=None)
    sp.set_defaults(fn=cmd_pull)

    sp = sub.add_parser("pull-checkpoint",
                        help="Download a published ORN checkpoint from HuggingFace")
    sp.add_argument("name", nargs="?", default=None,
                    help="e.g. orn-v3-605m. Omit or pass --list to see options.")
    sp.add_argument("--list", action="store_true")
    sp.add_argument("--cache", default=None)
    sp.set_defaults(fn=cmd_pull_checkpoint)

    # ── reproduce ──
    sp = sub.add_parser("reproduce", help="Run reproducible key results")
    sp.add_argument("names", nargs="*")
    sp.add_argument("--list", action="store_true")
    sp.add_argument("--slow", action="store_true", help="Include slow-tier results")
    sp.add_argument("--save-plots", default=None, metavar="DIR",
                    help="Save plots (PNG) for every result that defines plot()")
    sp.add_argument("--device", default=None)
    sp.set_defaults(fn=cmd_reproduce)

    args = p.parse_args()
    if not args.command:
        p.print_help(); sys.exit(1)
    args.fn(args)


if __name__ == "__main__":
    main()
