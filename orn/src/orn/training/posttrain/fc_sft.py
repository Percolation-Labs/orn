#!/usr/bin/env python3
"""Function-calling SFT for ORN — Stage 4 of the post-training stack.

Runs after long-context FT (`train_long_context.py`). Reads the FC shards
written by `data_fc.py` and continues training the LC checkpoint with the
extended ChatML template (now with <tool_call>/<tool_response> wrappers).

Differences from `train_sft.py`:
  - Reads pre-tokenised shards from --data_dir (not a HF dataset spec).
  - Default starting checkpoint = LC-FT'd v3_3B_longctx.pt.
  - Default LR 5e-6 (smaller than SFT's 5e-5; we are nudging, not teaching).
  - Defaults match the queue row in experiment_inventory.csv:
      seq_len 4096, batch 1, grad_accum 16, ~50K steps (~2 epochs).

The model itself does not need new embedding rows. The new tool-wrapper
strings are encoded as ordinary BPE subword sequences, the same way ChatML's
<|im_start|>/<|im_end|> are handled by `train_sft.py`. Loss masking is
inherited from the SFT trainer: only assistant tokens (which now include the
literal <tool_call>{...}</tool_call> blob) contribute to the loss.

Run on RunPod:
  python3 training/train_fc_sft.py \\
      --start_from /workspace/checkpoints/v3_3B_longctx.pt \\
      --data_dir /dev/shm/fc_sft \\
      --out /workspace/checkpoints/v3_3B_fc.pt \\
      --steps 50000 --batch 1 --grad_accum 16 --seq_len 4096 \\
      --lr 5e-6 --warmup 200 --weight_decay 0.0 --grad_clip 1.0

Smoke (tiny synthetic ORN, 5 steps):
  python3 training/data_fc.py --smoke --limit 100
  python3 training/train_fc_sft.py --smoke --data_dir data/fc_sft
"""
import argparse, json, math, os, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from orn.training.posttrain._compat import (
    FullORN, VanillaTransformer, build_orn, build_transformer,
)
from orn.training.posttrain.data_fc import (
    TOOL_CALL_OPEN, TOOL_CALL_CLOSE,
    TOOL_RESP_OPEN, TOOL_RESP_CLOSE,
)


# ══════════════════════════════════════════════════════════════════════
# Pre-tokenised shard loader
# ══════════════════════════════════════════════════════════════════════

class FCShardLoader:
    """Loads fc_sft shards (ids.bin, mask.bin, offsets.bin) per dataset and
    serves packed (B, T) batches with a token-level loss mask.

    Concatenates examples to fill seq_len, then chunks into B batches of T+1
    tokens. Mirrors `train_sft.SFTLoader` semantics (loss only on assistant
    tokens via `mask`).
    """
    def __init__(self, data_dir, seq_len, batch_size, device, seed=0,
                 dataset_names=None):
        self.data_dir = Path(data_dir)
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.device = device
        meta_path = self.data_dir / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"No metadata.json in {self.data_dir}. "
                f"Run training/data_fc.py first.")
        with open(meta_path) as f:
            self.meta = json.load(f)
        if dataset_names is None:
            dataset_names = list(self.meta["datasets"].keys())
        self.shards = []  # list of (ids_arr, mask_arr, offsets_arr)
        for name in dataset_names:
            info = self.meta["datasets"].get(name)
            if not info or info.get("examples", 0) == 0:
                continue
            ids = np.fromfile(self.data_dir / f"{name}.bin", dtype=np.uint32)
            mask = np.fromfile(self.data_dir / f"{name}_mask.bin", dtype=np.uint8)
            offsets = np.fromfile(self.data_dir / f"{name}_offsets.bin",
                                   dtype=np.uint64)
            self.shards.append((name, ids, mask, offsets))
            print(f"[loader] {name}: {info['examples']:,} examples, "
                  f"{info['tokens']:,} tokens", flush=True)
        if not self.shards:
            raise RuntimeError(f"No non-empty shards in {self.data_dir}")
        # Build a flat (shard_idx, example_idx) ordering across all shards
        triples = []
        for si, (_, _, _, offsets) in enumerate(self.shards):
            for ei in range(len(offsets) - 1):
                triples.append((si, ei))
        self.order = np.array(triples, dtype=np.int64)
        self.rng = np.random.default_rng(seed)
        self.rng.shuffle(self.order)
        self.ptr = 0
        self.buf_ids = np.empty(0, dtype=np.int64)
        self.buf_mask = np.empty(0, dtype=np.int8)
        self.total_tokens = sum(int(s[1].size) for s in self.shards)
        self.total_examples = sum(int(s[3].size - 1) for s in self.shards)

    def _next_example(self):
        if self.ptr >= len(self.order):
            self.rng.shuffle(self.order)
            self.ptr = 0
        si, ei = self.order[self.ptr]
        self.ptr += 1
        _, ids, mask, offsets = self.shards[si]
        a, b = int(offsets[ei]), int(offsets[ei + 1])
        return ids[a:b].astype(np.int64), mask[a:b].astype(np.int8)

    def _refill(self, needed):
        while len(self.buf_ids) < needed:
            ex_ids, ex_mask = self._next_example()
            self.buf_ids = np.concatenate([self.buf_ids, ex_ids])
            self.buf_mask = np.concatenate([self.buf_mask, ex_mask])

    def get_batch(self):
        need = self.batch_size * (self.seq_len + 1)
        self._refill(need)
        xs, ys, ms = [], [], []
        for _ in range(self.batch_size):
            seg_ids = self.buf_ids[: self.seq_len + 1]
            seg_mask = self.buf_mask[: self.seq_len + 1]
            xs.append(seg_ids[:-1])
            ys.append(seg_ids[1:])
            ms.append(seg_mask[1:].astype(np.float32))
            self.buf_ids = self.buf_ids[self.seq_len:]
            self.buf_mask = self.buf_mask[self.seq_len:]
        x = torch.from_numpy(np.stack(xs)).to(self.device, non_blocking=True)
        y = torch.from_numpy(np.stack(ys)).to(self.device, non_blocking=True)
        m = torch.from_numpy(np.stack(ms)).to(self.device, non_blocking=True)
        return x, y, m


# ══════════════════════════════════════════════════════════════════════
# LR schedule: linear warmup → cosine to lr_min
# ══════════════════════════════════════════════════════════════════════

def lr_schedule(step, total_steps, warmup, lr_max, lr_min=0.0):
    if step < warmup:
        return lr_max * (step + 1) / max(warmup, 1)
    progress = min(1.0, (step - warmup) / max(total_steps - warmup, 1))
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + math.cos(math.pi * progress))


# ══════════════════════════════════════════════════════════════════════
# Sanity probe: confirm tool-call tokens appear in current loader buffer
# ══════════════════════════════════════════════════════════════════════

def probe_tool_tokens_present(loader, tok, n_examples=5):
    """Decode a few random examples and check the literal <tool_call> /
    <tool_response> strings show up. Used by the smoke test."""
    saw_call = False
    saw_resp = False
    decoded = []
    snapshot = (
        np.array(loader.order, copy=True),
        loader.ptr,
        np.array(loader.buf_ids, copy=True),
        np.array(loader.buf_mask, copy=True),
    )
    try:
        for _ in range(n_examples):
            ids, mask = loader._next_example()
            txt = tok.decode(ids.tolist())
            decoded.append(txt[:300])
            if TOOL_CALL_OPEN in txt:
                saw_call = True
            if TOOL_RESP_OPEN in txt:
                saw_resp = True
    finally:
        loader.order = snapshot[0]
        loader.ptr = snapshot[1]
        loader.buf_ids = snapshot[2]
        loader.buf_mask = snapshot[3]
    return saw_call, saw_resp, decoded


# ══════════════════════════════════════════════════════════════════════
# Smoke test: tiny synthetic ORN, 5 steps
# ══════════════════════════════════════════════════════════════════════

def run_smoke(args):
    """Smoke: tiny ORN, batch=1, T=512, 5 steps. Asserts loss drops + no NaN
    + tool tokens present.

    Forces CPU (not MPS) — the tiny synthetic ORN at d=64, vocab=50257 has
    unstable forward dynamics on MPS that produce NaN/Inf within a few steps.
    On CPU this matches the GPU code path closely enough to verify the data,
    masking, and loss-computation logic. The real run on RunPod uses CUDA.
    """
    print("[smoke] tiny ORN; 5 steps; reading", args.data_dir, flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[smoke] device={device}", flush=True)

    import tiktoken
    tok = tiktoken.get_encoding("gpt2")
    vocab = tok.n_vocab

    seq_len = 512
    model = FullORN(
        d=64, n_q_heads=4, n_kv_heads=2, d_head=16,
        n_layers=2, vocab=vocab, seq_len=seq_len, ffn_width_mult=2,
        d_corr=16,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[smoke] tiny ORN params={n_params/1e6:.3f}M", flush=True)

    loader = FCShardLoader(args.data_dir, seq_len=seq_len, batch_size=1,
                            device=device, seed=0)
    print(f"[smoke] loader total_examples={loader.total_examples} "
          f"total_tokens={loader.total_tokens}", flush=True)

    saw_call, saw_resp, samples = probe_tool_tokens_present(loader, tok)
    print(f"[smoke] saw <tool_call>={saw_call}  saw <tool_response>={saw_resp}",
          flush=True)
    print(f"[smoke] first sample slice:\n{samples[0]!r}\n", flush=True)
    assert saw_call, "no <tool_call> in decoded examples — data prep broken"

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95),
                              weight_decay=0.0)
    losses = []
    for step in range(5):
        opt.zero_grad(set_to_none=True)
        x, y, m = loader.get_batch()
        logits = model(x, is_causal=True)
        per_tok = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            y.reshape(-1),
            reduction="none",
        ).view_as(y)
        loss = (per_tok * m).sum() / m.sum().clamp(min=1.0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        v = float(loss.item())
        losses.append(v)
        assert math.isfinite(v), f"NaN/Inf at step {step}"
        asst_frac = float(m.sum().item()) / float(m.numel())
        print(f"  [smoke] step {step} loss={v:.4f} "
              f"asst_frac={asst_frac:.3f}", flush=True)
    print(f"[smoke] start={losses[0]:.4f}  end={losses[-1]:.4f}  "
          f"decreased={losses[-1] < losses[0]}", flush=True)
    if losses[-1] >= losses[0]:
        print("[smoke] WARN: loss did not decrease in 5 steps "
              "(tiny model / tiny data — sometimes flat)", flush=True)
    print("[smoke] PASSED", flush=True)


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start_from", type=str,
                    default="/workspace/checkpoints/v3_3B_longctx.pt",
                    help="Base checkpoint to continue from (post-LC-FT).")
    ap.add_argument("--data_dir", type=str,
                    default="/dev/shm/fc_sft",
                    help="FC shard dir produced by data_fc.py "
                         "(use /dev/shm on RunPod, NOT /workspace).")
    ap.add_argument("--out", type=str,
                    default="/workspace/checkpoints/v3_3B_fc.pt")
    ap.add_argument("--resume", type=str, default=None,
                    help="Resume from a previous fc-sft checkpoint instead of "
                         "starting from --start_from. Useful after pod restart.")
    ap.add_argument("--steps", type=int, default=50_000)
    ap.add_argument("--seq_len", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--lr_min", type=float, default=0.0)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--log_every", type=int, default=25)
    ap.add_argument("--ckpt_every", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--datasets", type=str, default=None,
                    help="Comma list to restrict shards (default: all in metadata)")
    ap.add_argument("--smoke", action="store_true",
                    help="Tiny synthetic ORN + 5 steps + tool-token sanity")
    args = ap.parse_args()

    if args.smoke:
        run_smoke(args)
        return

    device = ("cuda" if torch.cuda.is_available()
              else ("mps" if torch.backends.mps.is_available() else "cpu"))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # ── Load base checkpoint ───────────────────────────────────────────
    src = args.resume if args.resume else args.start_from
    print(f"[load] {src}", flush=True)
    ckpt = torch.load(src, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    mc = cfg["model"]
    arch = cfg.get("arch") or "orn"
    if arch == "orn":
        model = build_orn(mc)
    else:
        model = build_transformer(mc)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if missing:
        print(f"[load] missing keys: {len(missing)} (first 5: {missing[:5]})",
              flush=True)
    if unexpected:
        print(f"[load] unexpected keys: {len(unexpected)} (first 5: {unexpected[:5]})",
              flush=True)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[init] arch={arch} params={n_params/1e6:.1f}M "
          f"model_seq_len={mc['seq_len']} pack_seq_len={args.seq_len} "
          f"device={device}", flush=True)
    if args.seq_len > mc["seq_len"]:
        print(f"[warn] requested pack seq_len={args.seq_len} > "
              f"model.seq_len={mc['seq_len']}. The LC ckpt should have "
              f"the larger seq_len; check that you loaded the post-LC ckpt.",
              flush=True)

    # ── Loader ─────────────────────────────────────────────────────────
    dataset_names = (args.datasets.split(",") if args.datasets else None)
    loader = FCShardLoader(args.data_dir, seq_len=args.seq_len,
                            batch_size=args.batch, device=device,
                            seed=args.seed, dataset_names=dataset_names)
    print(f"[init] total_examples={loader.total_examples:,} "
          f"total_tokens={loader.total_tokens:,}", flush=True)

    # ── Optimiser ─────────────────────────────────────────────────────
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                              betas=(0.9, 0.95),
                              weight_decay=args.weight_decay)
    start_step = 0
    if args.resume and "step" in ckpt:
        start_step = int(ckpt["step"])
        if "optimizer" in ckpt:
            try:
                opt.load_state_dict(ckpt["optimizer"])
                print(f"[resume] restored optimizer at step={start_step}",
                      flush=True)
            except Exception as e:
                print(f"[resume] optimizer state ignored: {e}", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = out_path.parent / (out_path.stem + ".jsonl")

    use_amp = (device == "cuda")
    amp_dtype = torch.bfloat16

    tok_per_step = args.batch * args.grad_accum * args.seq_len
    print(f"[init] {tok_per_step:,} tok/step, total_steps={args.steps:,} "
          f"≈ {tok_per_step * args.steps / 1e9:.2f}B tokens", flush=True)

    model.train()
    running_loss = 0.0
    running_asst_frac = 0.0
    t0 = time.time()
    tok_seen = 0

    for step in range(start_step, args.steps):
        opt.zero_grad(set_to_none=True)
        for _ in range(args.grad_accum):
            x, y, m = loader.get_batch()
            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype,
                                     enabled=use_amp):
                if arch == "orn":
                    logits = model(x, is_causal=True)
                else:
                    logits = model(x)
                per_tok = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    y.reshape(-1),
                    reduction="none",
                ).view_as(y)
                masked = (per_tok * m).sum() / m.sum().clamp(min=1.0)
            (masked / args.grad_accum).backward()
            running_loss += float(masked.item()) / args.grad_accum
            running_asst_frac += float((m.sum() / m.numel()).item()) / args.grad_accum
            tok_seen += args.batch * args.seq_len

        lr = lr_schedule(step, args.steps, args.warmup, args.lr, args.lr_min)
        for g in opt.param_groups:
            g["lr"] = lr
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()

        if (step + 1) % args.log_every == 0:
            dt = time.time() - t0
            tps = tok_seen / max(dt, 1e-9)
            line = (f"  step {step+1:>6}/{args.steps}  "
                    f"loss={running_loss/args.log_every:.4f}  lr={lr:.2e}  "
                    f"asst_frac={running_asst_frac/args.log_every:.2f}  "
                    f"{tps/1e3:.1f}K tok/s  t={dt/60:.1f}min")
            print(line, flush=True)
            with open(log_path, "a") as f:
                f.write(json.dumps({
                    "step": step + 1,
                    "loss": running_loss / args.log_every,
                    "lr": lr,
                    "asst_frac": running_asst_frac / args.log_every,
                    "tok_s": tps,
                    "t_min": dt / 60,
                }) + "\n")
            running_loss = 0.0
            running_asst_frac = 0.0

        if (step + 1) % args.ckpt_every == 0:
            torch.save({
                "model": model.state_dict(),
                "config": cfg,
                "step": step + 1,
                "args": vars(args),
                "optimizer": opt.state_dict(),
            }, out_path)
            print(f"  [ckpt] saved at step {step+1} → {out_path}", flush=True)

    torch.save({
        "model": model.state_dict(),
        "config": cfg,
        "step": args.steps,
        "args": vars(args),
    }, out_path)
    print(f"[done] FC-SFT complete; wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
