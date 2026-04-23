"""orn train — train a model from a JSON preset or path, with CLI overrides."""
from __future__ import annotations

from pathlib import Path


# Model-config overridable keys (forwarded into the model config block).
_MODEL_KEYS = (
    "d_model", "n_layers", "n_heads", "n_q_heads", "n_kv_heads",
    "d_head", "seq_len", "d_corr", "vocab_size", "ffn_width_mult",
)
# Training-config overridable keys (forwarded into the training config).
_TRAIN_KEYS = ("lr", "batch_size", "grad_accum_steps", "max_tokens", "warmup_steps")


def _ensure_data_dir(data_dir: str, config_name: str, seq: int,
                      batch_size: int, vocab_size: int) -> None:
    """For the dry_run config only, synthesise a small shard if data is missing.

    Lets a new user run `orn train --config dry_run` end-to-end with no HF
    token and no `orn prepare` call. Other configs are expected to use a
    properly prepared dataset.
    """
    import torch
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


def _apply_overrides(model_kw: dict, training_kw: dict, args) -> None:
    for k in _MODEL_KEYS:
        v = getattr(args, k, None)
        if v is not None:
            model_kw[k] = v
    for k in _TRAIN_KEYS:
        v = getattr(args, k, None)
        if v is not None:
            training_kw[k] = v
    if args.no_compile:         training_kw["compile"] = False
    if args.no_mixed_precision: training_kw["mixed_precision"] = False
    if args.output:             training_kw["output_dir"] = args.output


def cmd(args) -> None:
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
    _ensure_data_dir(data_dir, args.config, seq, train_cfg.batch_size,
                     vocab_size=model_kw.get("vocab_size", 50257))

    train_loader = ShardedDataLoader(data_dir, seq, train_cfg.batch_size, device)
    val_loader = ShardedDataLoader(data_dir, seq, train_cfg.batch_size, device, eval_mode=True)

    loss_fn = get_loss_fn(cfg.get("loss", {}).get("type", "lm_ce"))
    trainer = Trainer(model, train_cfg, train_loader, val_loader,
                      device=device, resume_from=args.resume, loss_fn=loss_fn)
    trainer.train()


def register(subparsers) -> None:
    sp = subparsers.add_parser("train", help="Train a model")
    sp.add_argument("--config", required=True, help="Preset name or path to JSON")
    sp.add_argument("--resume", default=None)
    sp.add_argument("--device", default=None)
    sp.add_argument("--data", default=None)
    sp.add_argument("--output", default=None)
    for k in _MODEL_KEYS:
        sp.add_argument(f"--{k.replace('_', '-')}", type=int, default=None)
    sp.add_argument("--lr", type=float, default=None)
    sp.add_argument("--batch-size", type=int, default=None)
    sp.add_argument("--grad-accum-steps", type=int, default=None)
    sp.add_argument("--max-tokens", type=int, default=None)
    sp.add_argument("--warmup-steps", type=int, default=None)
    sp.add_argument("--no-compile", action="store_true")
    sp.add_argument("--no-mixed-precision", action="store_true")
    sp.set_defaults(fn=cmd)
