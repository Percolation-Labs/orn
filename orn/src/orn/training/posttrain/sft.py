#!/usr/bin/env python3
"""Supervised fine-tuning for ORN (or vanilla transformer) with ChatML template.

Stage 1 of the 605M post-training stack (reports/training_plan_605m.md).

Loads a pretrain checkpoint, tokenises an SFT dataset with ChatML, masks
loss to assistant turn only, trains 1 epoch with LR 5e-5 cosine-to-zero.

Dataset: teknium/OpenHermes-2.5 by default (~1M dialogues, filterable).
ChatML format:
    <|im_start|>system
    {system}<|im_end|>
    <|im_start|>user
    {user}<|im_end|>
    <|im_start|>assistant
    {assistant}<|im_end|>

Loss is computed only on tokens inside the final assistant turn (masked
via loss_mask).

Run:
  python3 training/train_sft.py \\
      --pretrain /workspace/v3_latest.pt \\
      --dataset teknium/OpenHermes-2.5 \\
      --out /workspace/checkpoints/v3_3B_sft.pt \\
      --max_examples 400000 --max_len 2048 \\
      --lr 5e-5 --epochs 1 --batch 8 --grad_accum 4
"""
import argparse, json, math, os, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, "/workspace")
sys.path.insert(0, "/workspace/orn")

from orn.training.posttrain._compat import (
    FullORN, VanillaTransformer, build_orn, build_transformer,
)


# ══════════════════════════════════════════════════════════════════════
# ChatML template + loss-mask construction
# ══════════════════════════════════════════════════════════════════════

CHATML_START = "<|im_start|>"
CHATML_END = "<|im_end|>"


def format_chatml(messages):
    """messages: list of {"role": "system"|"user"|"assistant", "content": str}."""
    parts = []
    for m in messages:
        parts.append(f"{CHATML_START}{m['role']}\n{m['content']}{CHATML_END}\n")
    return "".join(parts)


def tokenise_with_mask(messages, tok, max_len, assistant_only_loss=True):
    """Return (ids, loss_mask) both 1-D lists.
    loss_mask[i] = 1 iff token i is part of an assistant turn (content only,
    not the <|im_start|>assistant preamble); else 0.
    """
    # We build token-by-token, tracking which segment each token belongs to.
    ids_list = []
    mask_list = []
    for m in messages:
        role = m["role"]
        preamble = f"{CHATML_START}{role}\n"
        body = m["content"]
        suffix = f"{CHATML_END}\n"
        pre_ids = tok.encode_ordinary(preamble)
        body_ids = tok.encode_ordinary(body)
        suf_ids = tok.encode_ordinary(suffix)
        ids_list.extend(pre_ids)
        mask_list.extend([0] * len(pre_ids))
        is_assistant = (role == "assistant")
        body_mask_val = (1 if (is_assistant or not assistant_only_loss) else 0)
        ids_list.extend(body_ids)
        mask_list.extend([body_mask_val] * len(body_ids))
        ids_list.extend(suf_ids)
        mask_list.extend([body_mask_val] * len(suf_ids))
    # Truncate to max_len
    ids_list = ids_list[:max_len]
    mask_list = mask_list[:max_len]
    return ids_list, mask_list


# ══════════════════════════════════════════════════════════════════════
# Dataset loading + normalisation
# ══════════════════════════════════════════════════════════════════════

def _norm_message(role, content):
    if isinstance(content, str):
        return {"role": role, "content": content}
    if isinstance(content, list):
        # HF sometimes has [{"type": "text", "text": ...}]; flatten
        return {"role": role, "content": "\n".join(
            str(c.get("text", c) if isinstance(c, dict) else c)
            for c in content
        )}
    return {"role": role, "content": str(content)}


def extract_messages(row):
    """Try common SFT dataset schemas, return list[{role, content}] or None."""
    # OpenHermes-2.5: "conversations": [{"from": "system|human|gpt", "value": ...}]
    if "conversations" in row and row["conversations"]:
        out = []
        for turn in row["conversations"]:
            frm = turn.get("from", "").lower()
            val = turn.get("value", "")
            if frm in ("system",):
                out.append(_norm_message("system", val))
            elif frm in ("human", "user"):
                out.append(_norm_message("user", val))
            elif frm in ("gpt", "assistant", "bot"):
                out.append(_norm_message("assistant", val))
        return out or None
    # UltraChat / generic: "messages": [{"role": ..., "content": ...}]
    if "messages" in row and row["messages"]:
        return [_norm_message(m.get("role", "user"), m.get("content", ""))
                for m in row["messages"]]
    # Alpaca-style prompt/response
    if "prompt" in row and "response" in row:
        return [_norm_message("user", row["prompt"]),
                _norm_message("assistant", row["response"])]
    return None


def stream_sft_examples(dataset_spec, tok, max_len,
                        max_examples, min_assistant_tokens=8):
    """Yield (ids, mask) for valid SFT examples up to max_examples.
    dataset_spec: "teknium/OpenHermes-2.5" etc.
    Tries common split names: train, train_sft, default.
    """
    from datasets import load_dataset
    ds = None
    for split in ("train", "train_sft", "default"):
        try:
            ds = load_dataset(dataset_spec, split=split, streaming=True)
            print(f"[data] loaded {dataset_spec} split={split}", flush=True)
            break
        except Exception as e:
            continue
    if ds is None:
        raise RuntimeError(f"Could not load {dataset_spec} under any known split")
    emitted = 0
    for row in ds:
        msgs = extract_messages(row)
        if not msgs:
            continue
        # Need at least one user + one assistant
        has_user = any(m["role"] == "user" for m in msgs)
        has_asst = any(m["role"] == "assistant" for m in msgs)
        if not (has_user and has_asst):
            continue
        ids, mask = tokenise_with_mask(msgs, tok, max_len)
        if len(ids) < 16:
            continue
        if sum(mask) < min_assistant_tokens:
            continue
        yield ids, mask
        emitted += 1
        if emitted >= max_examples:
            return


def tokenise_sft(out_dir, dataset_spec, max_examples, max_len, tok):
    """Tokenise and cache SFT examples to disk (npz with ids + mask)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = out_dir / "sft_packed.npz"
    if cache.exists():
        print(f"[data] using cached {cache}", flush=True)
        return cache
    ids_all = []
    mask_all = []
    lens = []
    start = time.time()
    for i, (ids, mask) in enumerate(stream_sft_examples(
            dataset_spec, tok, max_len, max_examples)):
        ids_all.append(np.array(ids, dtype=np.int32))
        mask_all.append(np.array(mask, dtype=np.int8))
        lens.append(len(ids))
        if (i + 1) % 5000 == 0:
            dt = time.time() - start
            print(f"  [data] {i+1:>7} examples, "
                  f"mean_len={np.mean(lens):.0f} "
                  f"mean_asst_mask={np.mean([m.sum() for m in mask_all[-5000:]]):.0f} "
                  f"t={dt:.0f}s", flush=True)
    # Save as ragged arrays (dtype=object for variable-length)
    np.savez(cache,
             ids=np.array(ids_all, dtype=object),
             mask=np.array(mask_all, dtype=object),
             lens=np.array(lens, dtype=np.int32))
    print(f"[data] wrote {len(ids_all)} examples to {cache}", flush=True)
    return cache


class SFTLoader:
    """Packed SFT loader. Concatenates tokens across examples to fill seq_len
    batches. Mask is preserved so loss is only on assistant tokens.
    """
    def __init__(self, cache_path, seq_len, batch_size, device, seed=0):
        data = np.load(cache_path, allow_pickle=True)
        self.ids_list = data["ids"]
        self.mask_list = data["mask"]
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.device = device
        self.rng = np.random.default_rng(seed)
        self.order = np.arange(len(self.ids_list))
        self.rng.shuffle(self.order)
        self.ptr = 0
        self.buf_ids = np.empty(0, dtype=np.int32)
        self.buf_mask = np.empty(0, dtype=np.int8)
        self.total_tokens = sum(len(x) for x in self.ids_list)
        self.epoch_tokens_seen = 0

    def _refill(self, needed):
        while len(self.buf_ids) < needed:
            if self.ptr >= len(self.order):
                self.rng.shuffle(self.order)
                self.ptr = 0
            idx = self.order[self.ptr]
            self.ptr += 1
            self.buf_ids = np.concatenate([self.buf_ids, self.ids_list[idx]])
            self.buf_mask = np.concatenate([self.buf_mask, self.mask_list[idx]])

    def get_batch(self):
        need = self.batch_size * (self.seq_len + 1)
        self._refill(need)
        xs, ys, ms = [], [], []
        for _ in range(self.batch_size):
            seg_ids = self.buf_ids[:self.seq_len + 1]
            seg_mask = self.buf_mask[:self.seq_len + 1]
            xs.append(seg_ids[:-1].astype(np.int64))
            ys.append(seg_ids[1:].astype(np.int64))
            ms.append(seg_mask[1:].astype(np.float32))  # mask on targets
            self.buf_ids = self.buf_ids[self.seq_len:]
            self.buf_mask = self.buf_mask[self.seq_len:]
        x = torch.from_numpy(np.stack(xs)).to(self.device, non_blocking=True)
        y = torch.from_numpy(np.stack(ys)).to(self.device, non_blocking=True)
        m = torch.from_numpy(np.stack(ms)).to(self.device, non_blocking=True)
        self.epoch_tokens_seen += self.batch_size * self.seq_len
        return x, y, m


# ══════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════

def cosine_to_zero_lr(step, total_steps, base_lr):
    if total_steps <= 0:
        return base_lr
    progress = min(1.0, step / total_steps)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrain", required=True, type=str,
                    help="Path to base checkpoint (.pt with config + model)")
    ap.add_argument("--dataset", default="teknium/OpenHermes-2.5",
                    type=str)
    ap.add_argument("--out", required=True, type=str,
                    help="Output SFT checkpoint path (.pt)")
    ap.add_argument("--cache_dir", default="/workspace/sft_cache", type=str)
    ap.add_argument("--max_examples", type=int, default=400_000)
    ap.add_argument("--max_len", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--device", default=None)
    ap.add_argument("--pack_len", type=int, default=None,
                    help="Override loader's pack length (model seq_len stays "
                         "unchanged; shorter packed sequences save activation "
                         "memory on wide-FFN models). Use 1024 or 512 if OOM at 2048.")
    args = ap.parse_args()

    if args.device is None:
        args.device = ("cuda" if torch.cuda.is_available()
                       else ("mps" if torch.backends.mps.is_available() else "cpu"))

    # Load base checkpoint
    print(f"[load] pretrain={args.pretrain}", flush=True)
    ckpt = torch.load(args.pretrain, map_location=args.device, weights_only=False)
    cfg = ckpt["config"]
    mc = cfg["model"]
    arch = cfg.get("arch") or "orn"
    pack_len = args.pack_len or mc["seq_len"]
    print(f"[load] arch={arch} val_loss={ckpt.get('val_loss')} "
          f"tokens_seen={ckpt.get('tokens_seen')}", flush=True)

    # Build + load
    if arch == "orn":
        model = build_orn(mc)
    else:
        model = build_transformer(mc)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if missing:
        print(f"[load] missing keys: {len(missing)} (first 5: {missing[:5]})", flush=True)
    if unexpected:
        print(f"[load] unexpected keys: {len(unexpected)} (first 5: {unexpected[:5]})", flush=True)
    model = model.to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[init] arch={arch} params={n_params/1e6:.1f}M "
          f"seq_len={mc['seq_len']} device={args.device}", flush=True)

    # Tokenise SFT dataset
    import tiktoken
    tok = tiktoken.get_encoding("gpt2")
    cache_path = tokenise_sft(
        Path(args.cache_dir), args.dataset, args.max_examples,
        args.max_len, tok,
    )

    loader = SFTLoader(cache_path, pack_len, args.batch, args.device, seed=42)
    print(f"[init] pack_len={pack_len} (model seq_len={mc['seq_len']})", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                             betas=(0.9, 0.95),
                             weight_decay=args.weight_decay)

    tok_per_step = args.batch * args.grad_accum * pack_len
    total_steps = args.epochs * (loader.total_tokens // tok_per_step)
    print(f"[init] {loader.total_tokens:,} total tokens, "
          f"{tok_per_step:,} tok/step, {total_steps:,} total steps", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = out_path.parent / "sft_train.jsonl"

    model.train()
    step = 0
    running_loss = 0.0
    running_asst_frac = 0.0
    t0 = time.time()
    tok_seen = 0

    # bf16 autocast on A100/H100; fp32 optimizer states
    use_amp = (args.device == "cuda")
    amp_dtype = torch.bfloat16

    while step < total_steps:
        opt.zero_grad(set_to_none=True)
        for _ in range(args.grad_accum):
            x, y, m = loader.get_batch()
            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                if arch == "orn":
                    logits = model(x, is_causal=True)
                else:
                    logits = model(x)
                # Compute per-token loss then mask to assistant tokens only
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    y.reshape(-1),
                    reduction="none",
                ).view_as(y)
                mask_sum = m.sum().clamp(min=1.0)
                masked_loss = (loss * m).sum() / mask_sum
            (masked_loss / args.grad_accum).backward()
            running_loss += float(masked_loss.item()) / args.grad_accum
            running_asst_frac += float((m.sum() / m.numel()).item()) / args.grad_accum
            tok_seen += args.batch * pack_len

        # LR schedule
        lr = cosine_to_zero_lr(step, total_steps, args.lr)
        for g in opt.param_groups:
            g["lr"] = lr
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        step += 1

        if step % args.log_every == 0:
            dt = time.time() - t0
            throughput = tok_seen / max(dt, 1e-9)
            line = (f"  step {step:>5}/{total_steps}  loss={running_loss/args.log_every:.4f}"
                    f"  lr={lr:.2e}  asst_frac={running_asst_frac/args.log_every:.2f}"
                    f"  {throughput/1e3:.1f}K tok/s  t={dt/60:.1f}min")
            print(line, flush=True)
            with open(log_path, "a") as f:
                f.write(json.dumps({
                    "step": step, "loss": running_loss/args.log_every,
                    "lr": lr, "asst_frac": running_asst_frac/args.log_every,
                    "tok_s": throughput, "t_min": dt/60,
                }) + "\n")
            running_loss = 0.0
            running_asst_frac = 0.0

        if step % args.eval_every == 0:
            # Quick save
            torch.save({
                "model": model.state_dict(),
                "config": cfg,
                "step": step,
                "args": vars(args),
            }, out_path)
            print(f"  [ckpt] saved at step {step}", flush=True)

    # Final save
    torch.save({
        "model": model.state_dict(),
        "config": cfg,
        "step": step,
        "args": vars(args),
    }, out_path)
    print(f"[done] SFT complete; wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
