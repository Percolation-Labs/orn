#!/usr/bin/env python3
"""Direct Preference Optimization (DPO) for ORN.

Stage 3 of the 605M post-training stack (reports/training_plan_605m.md).

Loads a post-SFT checkpoint (the policy), creates a frozen copy as the
reference, and trains the policy on preference pairs (chosen/rejected)
with the DPO loss:

    L = -log sigma( beta * (log pi(chosen) - log pi_ref(chosen)
                          - log pi(rejected) + log pi_ref(rejected)) )

Log probabilities are summed over the RESPONSE tokens only (prompt tokens
are masked). Both chosen and rejected use the same prompt.

Dataset: HuggingFaceH4/ultrafeedback_binarized by default. Schema:
  {prompt, chosen: [{role,content},...], rejected: [{role,content},...]}

Run:
  python3 training/train_dpo.py \\
      --policy /workspace/checkpoints/v3_3B_sft.pt \\
      --dataset HuggingFaceH4/ultrafeedback_binarized \\
      --out /workspace/checkpoints/v3_3B_dpo.pt \\
      --beta 0.1 --lr 5e-7 --epochs 2 --batch 2 --grad_accum 8 \\
      --max_examples 60000 --pack_len 1024
"""
import argparse, copy, json, math, os, sys, time
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
# Preference-pair extraction + tokenisation
# ══════════════════════════════════════════════════════════════════════
#
# UltraFeedback binarized row: {chosen: [...], rejected: [...], prompt: "..."}
# Both chosen and rejected are ChatML-style conversation lists with the
# final assistant turn differing between them.

def extract_chosen_rejected(row):
    """Return (prompt_str, chosen_response_str, rejected_response_str)
    or None if schema doesn't match.
    """
    chosen = row.get("chosen")
    rejected = row.get("rejected")
    if not chosen or not rejected:
        return None
    # chosen/rejected should be lists of {role, content}
    if not (isinstance(chosen, list) and isinstance(rejected, list)):
        return None

    def extract_prompt_and_reply(conv):
        # Prompt = all turns except the final assistant
        # Reply = final assistant's content
        if not conv:
            return None, None
        if conv[-1].get("role") != "assistant":
            return None, None
        prompt_turns = conv[:-1]
        reply = conv[-1].get("content", "")
        return prompt_turns, reply

    pc, rc = extract_prompt_and_reply(chosen)
    pr, rr = extract_prompt_and_reply(rejected)
    if pc is None or pr is None or not rc or not rr:
        return None
    # Verify prompts match (they should)
    if pc != pr:
        # Not a true preference pair; skip
        return None
    return pc, rc, rr


def tokenise_pair(prompt_turns, chosen, rejected, tok, max_len):
    """Return (prompt_ids, chosen_ids, rejected_ids) with ChatML formatting.
    prompt_ids: formatted prompt (system+user turns) ending in the assistant
      preamble. This is common to both sides.
    chosen_ids / rejected_ids: the response tokens (+ end-of-turn marker).
    """
    # Build prompt string: each turn <|im_start|>role\ncontent<|im_end|>\n,
    # then <|im_start|>assistant\n (for generation)
    prompt_str = ""
    for m in prompt_turns:
        r = m.get("role", "user")
        c = m.get("content", "")
        prompt_str += f"{CHATML_START}{r}\n{c}{CHATML_END}\n"
    prompt_str += f"{CHATML_START}assistant\n"

    end_marker = f"{CHATML_END}\n"

    prompt_ids = tok.encode_ordinary(prompt_str)
    chosen_ids = tok.encode_ordinary(chosen + end_marker)
    rejected_ids = tok.encode_ordinary(rejected + end_marker)

    # Truncate: keep prompt + response within max_len. If prompt alone
    # exceeds max_len-16, skip.
    if len(prompt_ids) > max_len - 16:
        return None
    budget_chosen = max_len - len(prompt_ids)
    budget_rejected = max_len - len(prompt_ids)
    chosen_ids = chosen_ids[:budget_chosen]
    rejected_ids = rejected_ids[:budget_rejected]
    if not chosen_ids or not rejected_ids:
        return None
    return prompt_ids, chosen_ids, rejected_ids


def stream_dpo_pairs(dataset_spec, tok, max_len, max_examples):
    from datasets import load_dataset
    ds = None
    for split in ("train_prefs", "train", "train_sft", "default"):
        try:
            ds = load_dataset(dataset_spec, split=split, streaming=True)
            print(f"[data] loaded {dataset_spec} split={split}", flush=True)
            break
        except Exception:
            continue
    if ds is None:
        raise RuntimeError(f"Could not load {dataset_spec}")
    emitted = 0
    for row in ds:
        extracted = extract_chosen_rejected(row)
        if extracted is None:
            continue
        prompt_turns, chosen, rejected = extracted
        tok_out = tokenise_pair(prompt_turns, chosen, rejected, tok, max_len)
        if tok_out is None:
            continue
        p_ids, c_ids, r_ids = tok_out
        yield p_ids, c_ids, r_ids
        emitted += 1
        if emitted >= max_examples:
            return


def tokenise_and_cache(cache_dir, dataset_spec, max_examples, max_len, tok):
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / "dpo_packed.npz"
    if cache.exists():
        print(f"[data] using cached {cache}", flush=True)
        return cache
    p_all, c_all, r_all = [], [], []
    t0 = time.time()
    for i, (p, c, r) in enumerate(stream_dpo_pairs(
            dataset_spec, tok, max_len, max_examples)):
        p_all.append(np.array(p, dtype=np.int32))
        c_all.append(np.array(c, dtype=np.int32))
        r_all.append(np.array(r, dtype=np.int32))
        if (i + 1) % 2000 == 0:
            print(f"  [data] {i+1:>6} pairs tokenised t={time.time()-t0:.0f}s",
                  flush=True)
    np.savez(cache,
             p=np.array(p_all, dtype=object),
             c=np.array(c_all, dtype=object),
             r=np.array(r_all, dtype=object))
    print(f"[data] wrote {len(p_all)} pairs to {cache}", flush=True)
    return cache


class DPOLoader:
    def __init__(self, cache_path, pack_len, batch_size, device, seed=0):
        data = np.load(cache_path, allow_pickle=True)
        self.p = data["p"]; self.c = data["c"]; self.r = data["r"]
        self.pack_len = pack_len
        self.batch_size = batch_size
        self.device = device
        self.rng = np.random.default_rng(seed)
        self.order = np.arange(len(self.p))
        self.rng.shuffle(self.order)
        self.ptr = 0

    def _get_one(self):
        if self.ptr >= len(self.order):
            self.rng.shuffle(self.order)
            self.ptr = 0
        idx = self.order[self.ptr]
        self.ptr += 1
        p = self.p[idx]
        c = self.c[idx]
        r = self.r[idx]
        # Build full sequences: prompt + response, padded/truncated to pack_len
        def concat_pad(prompt, resp):
            ids = np.concatenate([prompt, resp])[:self.pack_len]
            # response mask: 1 for response tokens, 0 for prompt
            mask = np.zeros(len(ids), dtype=np.float32)
            r_start = len(prompt)
            mask[r_start:] = 1.0
            # Pad to pack_len
            if len(ids) < self.pack_len:
                pad = self.pack_len - len(ids)
                pad_tok = ids[-1] if len(ids) else 0
                ids = np.concatenate([ids, np.full(pad, pad_tok, dtype=np.int32)])
                mask = np.concatenate([mask, np.zeros(pad, dtype=np.float32)])
            return ids, mask
        cids, cmask = concat_pad(p, c)
        rids, rmask = concat_pad(p, r)
        return cids, cmask, rids, rmask

    def get_batch(self):
        cids, cmasks, rids, rmasks = [], [], [], []
        for _ in range(self.batch_size):
            ci, cm, ri, rm = self._get_one()
            cids.append(ci); cmasks.append(cm)
            rids.append(ri); rmasks.append(rm)
        c_ids = torch.from_numpy(np.stack(cids).astype(np.int64)).to(self.device)
        c_mask = torch.from_numpy(np.stack(cmasks)).to(self.device)
        r_ids = torch.from_numpy(np.stack(rids).astype(np.int64)).to(self.device)
        r_mask = torch.from_numpy(np.stack(rmasks)).to(self.device)
        return c_ids, c_mask, r_ids, r_mask


# ══════════════════════════════════════════════════════════════════════
# DPO loss
# ══════════════════════════════════════════════════════════════════════

def log_probs_response(model, ids, resp_mask, arch, amp_dtype, use_amp=True):
    """Sum log-prob over response tokens only (where resp_mask==1).
    ids:       (B, S) -- input token ids
    resp_mask: (B, S) -- 1 for response tokens, 0 elsewhere. NOTE: resp_mask
               is position-i=1 if position i is a response TARGET; we shift
               the mask by 1 to align with next-token prediction.
    Returns (B,) sum of log probs over response tokens.
    """
    x = ids[:, :-1]
    y = ids[:, 1:]
    mask = resp_mask[:, 1:]  # aligned with targets
    with torch.amp.autocast(device_type="cuda" if x.is_cuda else "cpu",
                             dtype=amp_dtype, enabled=use_amp):
        if arch == "orn":
            logits = model(x, is_causal=True)
        else:
            logits = model(x)
        log_probs = F.log_softmax(logits.float(), dim=-1)
        # Gather log-prob at target y
        lp_at_y = log_probs.gather(-1, y.unsqueeze(-1)).squeeze(-1)  # (B, S-1)
    lp_sum = (lp_at_y * mask).sum(dim=-1)  # (B,)
    return lp_sum


def dpo_loss(pi_c, pi_r, ref_c, ref_r, beta=0.1):
    """DPO loss. Inputs are (B,) sums of log probs.
      pi_c:  log p_policy(chosen)
      pi_r:  log p_policy(rejected)
      ref_c: log p_ref(chosen)
      ref_r: log p_ref(rejected)
    """
    logits = beta * ((pi_c - ref_c) - (pi_r - ref_r))
    loss = -F.logsigmoid(logits).mean()
    # Also compute reward margin for logging
    reward_margin = ((pi_c - ref_c) - (pi_r - ref_r)).mean().item()
    reward_acc = (logits > 0).float().mean().item()
    return loss, reward_margin, reward_acc


def cosine_lr(step, total, base, min_lr=None):
    if total <= 0:
        return base
    if min_lr is None:
        min_lr = 0.0
    progress = min(1.0, step / total)
    return min_lr + (base - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True,
                    help="Post-SFT checkpoint to use as both policy and ref.")
    ap.add_argument("--dataset", default="HuggingFaceH4/ultrafeedback_binarized",
                    type=str)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache_dir", default="/workspace/dpo_cache", type=str)
    ap.add_argument("--max_examples", type=int, default=60_000)
    ap.add_argument("--pack_len", type=int, default=1024)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=5e-7)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--log_every", type=int, default=25)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    if args.device is None:
        args.device = ("cuda" if torch.cuda.is_available()
                       else ("mps" if torch.backends.mps.is_available() else "cpu"))

    print(f"[load] policy + ref from {args.policy}", flush=True)
    ckpt = torch.load(args.policy, map_location=args.device, weights_only=False)
    cfg = ckpt["config"]; mc = cfg["model"]
    arch = cfg.get("arch") or "orn"

    # Build policy
    if arch == "orn":
        policy = build_orn(mc)
    else:
        policy = build_transformer(mc)
    policy.load_state_dict(ckpt["model"], strict=False)
    policy = policy.to(args.device)

    # Build reference (frozen copy)
    if arch == "orn":
        ref = build_orn(mc)
    else:
        ref = build_transformer(mc)
    ref.load_state_dict(ckpt["model"], strict=False)
    ref = ref.to(args.device).eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    n_params = sum(p.numel() for p in policy.parameters())
    print(f"[init] arch={arch} params={n_params/1e6:.1f}M seq_len={mc['seq_len']}",
          flush=True)

    import tiktoken
    tok = tiktoken.get_encoding("gpt2")

    cache_path = tokenise_and_cache(
        Path(args.cache_dir), args.dataset, args.max_examples, args.max_len, tok,
    )
    loader = DPOLoader(cache_path, args.pack_len, args.batch, args.device, seed=42)

    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr,
                             betas=(0.9, 0.95),
                             weight_decay=args.weight_decay)

    tok_per_step = args.batch * args.grad_accum * args.pack_len
    n_pairs = len(loader.p)
    total_steps = args.epochs * (n_pairs // (args.batch * args.grad_accum))
    print(f"[init] {n_pairs} pairs, {tok_per_step:,} tok/step, "
          f"{total_steps:,} total steps",
          flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = out_path.parent / "dpo_train.jsonl"

    use_amp = (args.device == "cuda")
    amp_dtype = torch.bfloat16

    policy.train()
    step = 0
    running_loss, running_margin, running_acc = 0.0, 0.0, 0.0
    t0 = time.time(); tok_seen = 0

    while step < total_steps:
        opt.zero_grad(set_to_none=True)
        for _ in range(args.grad_accum):
            c_ids, c_mask, r_ids, r_mask = loader.get_batch()
            # Reference log-probs (no grad)
            with torch.no_grad():
                ref_c = log_probs_response(ref, c_ids, c_mask, arch,
                                            amp_dtype, use_amp)
                ref_r = log_probs_response(ref, r_ids, r_mask, arch,
                                            amp_dtype, use_amp)
            # Policy log-probs (with grad)
            pi_c = log_probs_response(policy, c_ids, c_mask, arch,
                                        amp_dtype, use_amp)
            pi_r = log_probs_response(policy, r_ids, r_mask, arch,
                                        amp_dtype, use_amp)
            loss, margin, acc = dpo_loss(pi_c, pi_r, ref_c, ref_r,
                                           beta=args.beta)
            (loss / args.grad_accum).backward()
            running_loss += float(loss.item()) / args.grad_accum
            running_margin += margin / args.grad_accum
            running_acc += acc / args.grad_accum
            tok_seen += args.batch * args.pack_len * 2  # chosen + rejected

        lr = cosine_lr(step, total_steps, args.lr)
        for g in opt.param_groups:
            g["lr"] = lr
        torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
        opt.step()
        step += 1

        if step % args.log_every == 0:
            dt = time.time() - t0
            throughput = tok_seen / max(dt, 1e-9)
            line = (f"  step {step:>5}/{total_steps}  "
                    f"loss={running_loss/args.log_every:.4f}  "
                    f"margin={running_margin/args.log_every:+.3f}  "
                    f"acc={running_acc/args.log_every:.2f}  "
                    f"lr={lr:.2e}  "
                    f"{throughput/1e3:.1f}K tok/s  t={dt/60:.1f}min")
            print(line, flush=True)
            with open(log_path, "a") as f:
                f.write(json.dumps({
                    "step": step, "loss": running_loss/args.log_every,
                    "margin": running_margin/args.log_every,
                    "acc": running_acc/args.log_every,
                    "lr": lr, "tok_s": throughput, "t_min": dt/60,
                }) + "\n")
            running_loss = 0.0; running_margin = 0.0; running_acc = 0.0

        if step % args.eval_every == 0:
            torch.save({
                "model": policy.state_dict(),
                "config": cfg,
                "step": step,
                "args": vars(args),
            }, out_path)
            print(f"  [ckpt] saved at step {step}", flush=True)

    torch.save({
        "model": policy.state_dict(),
        "config": cfg,
        "step": step,
        "args": vars(args),
    }, out_path)
    print(f"[done] DPO complete; wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
