"""orn predict — generate text from a checkpoint (local path or known name)."""
from __future__ import annotations


def cmd(args) -> None:
    import torch
    from orn.utils import load_checkpoint, pick_device, resolve_local, load_known
    from orn.utils.hf import get_tokenizer
    from orn.models import build_model

    device = pick_device(args.device)
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
            raise ValueError(
                "Checkpoint has no model_config.arch. Re-train with the current "
                "trainer, or use a known checkpoint name (orn pull-checkpoint --list)."
            )
        model = build_model(arch, model_cfg)
        model.load_state_dict(ckpt["model"])
        model = model.to(device).eval()

    enc = get_tokenizer(args.tokenizer)
    prompt = args.prompt or "The"
    ids = torch.tensor(
        enc.encode(prompt) if hasattr(enc, "encode") else enc(prompt)["input_ids"],
        device=device,
    )
    out = model.generate(ids, max_new=args.max_tokens,
                         temperature=args.temperature, top_k=args.top_k)
    print(enc.decode(out.tolist()))


def register(subparsers) -> None:
    sp = subparsers.add_parser("predict", help="Generate text from a checkpoint")
    sp.add_argument("--checkpoint", required=True,
                    help="Local path or known name (orn-v2-108m, orn-v3-605m, ...)")
    sp.add_argument("--prompt", default="The")
    sp.add_argument("--max-tokens", type=int, default=200)
    sp.add_argument("--temperature", type=float, default=0.8)
    sp.add_argument("--top-k", type=int, default=40)
    sp.add_argument("--device", default=None)
    sp.add_argument("--tokenizer", default="tiktoken-gpt2")
    sp.set_defaults(fn=cmd)
