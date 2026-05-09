#!/usr/bin/env python3
"""Long-context fine-tune for the DPO'd (or SFT'd) ORN — PoSE-correct.

═══════════════════════════════════════════════════════════════════════════
                          WHAT THIS SCRIPT DOES
═══════════════════════════════════════════════════════════════════════════

Takes a post-training checkpoint trained at modest T (typically T=1024) and
extends it to a long-context regime via:

  1. NTK-aware RoPE base rescaling  (Peng et al. 2023, "NTK-aware Scaled
     RoPE"). The original RoPE base θ=10000 is rescaled to
         θ_new = θ * (T_new/T_old) ^ (D / (D - 2))
     so that *all frequency bands* extrapolate smoothly to the new range,
     not just the high-frequency ones. This avoids the YaRN attention-
     entropy collapse at long T.

  2. PoSE position sampling  (Zhu et al. 2024, ICLR). Train at small
     T_content (e.g. 2048) but with positions sampled in [0, T_virtual)
     where T_virtual is the target eval length. Each batch element gets
     n_chunks of contiguous-position chunks with random offsets — the
     model sees diverse position pairs without paying O(T^2) memory.

  3. Standard NTP loss on FineWeb-edu (or whatever data shards). Same
     LR as DPO (~5e-7) so we don't undo the DPO-shifted preferences.

Result: a model whose RoPE schedule is calibrated for T=T_virtual without
ever forwarding a long sequence at training time.

═══════════════════════════════════════════════════════════════════════════
                CONNECTION TO THE LORN R-INJECTION WORK
═══════════════════════════════════════════════════════════════════════════

This trainer is the LARGE-MODEL companion to the small-model rotation-
injection experiments. The two routes share the same mathematical operation
— modulating L's attention bilinear via per-position rotations on K — but
take it from different sources:

   small model (LORN v3):  rotations come from a co-trained R module
                           reading the same token stream.
   large model (this):     rotations come from L's own RoPE seeing
                           PoSE-sampled positions in [0, T_virtual).

If our Appendix-D claim is right ("rotation is rotation, regardless of
where it comes from"), then:

   • A 3B fine-tuned this way and a 16M with co-trained R should produce
     comparable C2 long-context-by-proxy advantages, when normalized by
     model scale.
   • The rotational gauge that emerges in the 3B's pretraining-frozen
     weights via this fine-tune should be predictable from the small-
     model's R-emitted gauge — the channels that R learned to twist most
     should align (modulo basis) with the channels that PoSE-induced
     long-context training pushes the 3B toward.

Specifically informed BY the R work:
   • bf16 RoPE precision fix (fp32 trig tables) — inherited from
     orn/rope_pose.py. The DPO'd 3B otherwise loses precision in
     sin(m·θ) at large m.
   • Frequency-band sensitivity (YaRN's NTK-by-parts intuition): in
     the small-model R sweep we found low-band rotations are gentler
     than high-band ones. Here we apply NTK-aware scaling which acts
     on all bands but most strongly on the low-frequency end.
   • PoSE positions actually flowing through to RoPE (the bug we fixed
     in the small-model trainer; reusing the apply_rope_pos from
     orn/rope_pose.py for free).

Specifically informing the R work:
   • The 3B's loss curve under PoSE-correct training gives a *production-
     scale baseline* for the C2 number. If R-injection at 16M produces
     a smaller proxy_advantage than this 3B's natural long-context
     advantage at matched T_eval, R-injection is a smaller-but-cheaper
     substitute. If the numbers match, the architectures are equivalent
     at the rotational-channel level.
   • The 3B's emergent attention temperature shifts at long T (via the
     softmax denominator integrating over more tokens) tell us what
     R *should* be doing in the small model — we can use the 3B's
     post-FT RoPE/attention profile as a target signal for R.
   • Whether this 3B FT works at all (positive C2) is itself evidence
     that the rotational-gauge family of operations is sufficient for
     long-context calibration. If it fails for the 3B at this scale,
     the entire "rotation is the right operation" thesis takes damage.

═══════════════════════════════════════════════════════════════════════════
                                  USAGE
═══════════════════════════════════════════════════════════════════════════

After DPO completes on RunPod (~11:00 UTC 2026-04-26):

  python3 training/train_long_context.py \\
      --policy /workspace/checkpoints/v3_3B_dpo.pt \\
      --data_dir /dev/shm/fineweb \\
      --out /workspace/checkpoints/v3_3B_longctx.pt \\
      --T_content 2048 --T_virtual 16384 --n_chunks 4 \\
      --steps 10000 --batch 1 --grad_accum 16 \\
      --lr 5e-7 --warmup 200

Smoke test (tiny model, no real checkpoint required):

  python3 training/train_long_context.py --smoke
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

# We use FullORN (the L architecture) and the position-aware RoPE we wrote
# for the small-model PoSE fix. Critically, we re-use apply_rope_pos so the
# bug fix from the small-model trainer carries forward at zero cost.
from orn.training.posttrain._compat import FullORN, build_orn
from orn.models.layers import precompute_rope_freqs

# PoSE-aware RoPE + sharded long-context loader live in the private repo
# at this scale-out tier. The public package keeps placeholders so the
# import doesn't break; if you actually want to run long-context FT,
# port `orn.rope_pose` and `pose_loader` from the private repo into here.
try:
    from orn.rope_pose import apply_rope_pos  # type: ignore[import-not-found]
except ImportError:
    apply_rope_pos = None  # type: ignore[assignment]
try:
    from orn.training.posttrain.pose_loader import PoSELongShardLoader  # type: ignore[import-not-found]
except ImportError:
    PoSELongShardLoader = None  # type: ignore[assignment]


# ══════════════════════════════════════════════════════════════════════
# NTK-aware RoPE base rescaling
# ══════════════════════════════════════════════════════════════════════

def ntk_aware_theta(theta_old: float, T_old: int, T_new: int, d_head: int) -> float:
    """Compute the NTK-aware-rescaled RoPE base.

    Per Peng et al. 2023 ("NTK-aware Scaled RoPE"), the appropriate
    rescaling that interpolates ALL frequency bands smoothly to the new
    context length is:

        θ_new = θ_old * (T_new / T_old) ^ (D / (D - 2))

    where D is d_head. The exponent D/(D-2) ensures the highest-frequency
    band stays approximately intact (so local position resolution is
    preserved) while the lowest-frequency bands stretch to span the new
    range. With D=64 this is 1.032 — a ~3% boost above pure linear
    scaling. With D=128 it's 1.016 — even closer to linear.
    """
    scale = (T_new / T_old) ** (d_head / (d_head - 2))
    return theta_old * scale


def rebuild_rope_freqs(model: FullORN, T_virtual: int, theta_new: float,
                       device: str) -> None:
    """Replace the model's rope_freqs buffer with one for the new T_virtual
    range and rescaled theta. Done in-place on the buffer so that all blocks
    (which hold references to the same buffer) automatically pick up the
    new table."""
    d_head = model.blocks[0].d_head
    new_freqs = precompute_rope_freqs(d_head, T_virtual + 16, theta=theta_new)
    # Replace buffer in-place. All ORNBlock instances received the same
    # buffer reference at construction time, so updating model.rope_freqs
    # propagates to every block.
    model.rope_freqs = new_freqs.to(device)
    for blk in model.blocks:
        blk.rope_freqs = new_freqs.to(device)


# ══════════════════════════════════════════════════════════════════════
# PoSE-aware forward path
# ══════════════════════════════════════════════════════════════════════

def block_forward_pose(block, x, positions):
    """Drop-in replacement for ORNBlock.forward that uses PoSE positions.
    Mirrors the small-model `block_forward_pose` in train_lorn_v3_pose.py
    but without the R-rotation injection (no R in this trainer)."""
    B, S, D = x.shape
    h = block.ln_attn(x)

    q = (h @ block.A).view(B, S, block.n_q_heads, block.d_head).transpose(1, 2)
    k_raw = h @ block.B
    k = block.Wk_proj(k_raw).view(B, S, block.n_kv_heads, block.d_head).transpose(1, 2)
    v = block.Wv(h).view(B, S, block.n_kv_heads, block.d_head).transpose(1, 2)

    # Position-aware RoPE — the ONE line that distinguishes this from the
    # default ORNBlock.forward.
    q = apply_rope_pos(q, block.rope_freqs.to(q.device), positions)
    k = apply_rope_pos(k, block.rope_freqs.to(k.device), positions)

    k = k.repeat_interleave(block.n_rep, dim=1)
    v = v.repeat_interleave(block.n_rep, dim=1)

    attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    attn = attn.transpose(1, 2).contiguous().view(B, S, -1)
    attn = block.Wo(attn)
    x = x + attn

    x = x + block.shared_ffn(block.ln_ffn(x))
    h_c = block.ln_corr(x)
    x = x + torch.sigmoid(block.gate(h_c)) * block.corr(h_c)
    return x


def L_forward_pose(L, tokens, positions, targets=None):
    """Position-aware NTP forward."""
    B, S = tokens.shape
    h = L.tok_emb(tokens)
    for block in L.blocks:
        h = block_forward_pose(block, h, positions)
    h = L.ln_f(h)
    logits = L.head(h)
    if targets is not None:
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            reduction="none",
        ).reshape(B, S)
    else:
        loss = None
    return logits, loss


# ══════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", help="Path to checkpoint to fine-tune "
                                       "(e.g. v3_3B_dpo.pt)")
    ap.add_argument("--data_dir", help="FineWeb-edu shard dir")
    ap.add_argument("--out", help="Output checkpoint path")
    ap.add_argument("--T_content", type=int, default=2048,
                    help="Per-batch content length")
    ap.add_argument("--T_virtual", type=int, default=16384,
                    help="Target virtual position range")
    ap.add_argument("--n_chunks", type=int, default=4,
                    help="PoSE chunk count")
    ap.add_argument("--T_train_old", type=int, default=1024,
                    help="Original training T (for NTK rescaling)")
    ap.add_argument("--steps", type=int, default=10_000)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-7,
                    help="Use small LR (post-DPO continuation)")
    ap.add_argument("--lr_min", type=float, default=5e-8)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--log_every", type=int, default=25)
    ap.add_argument("--ckpt_every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true",
                    help="Run a tiny synthetic smoke test (no real ckpt).")
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available()
              else ("mps" if torch.backends.mps.is_available() else "cpu"))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # ────────────────────────────────────────────────────────────────
    # Smoke test path: tiny synthetic model, tiny synthetic data
    # ────────────────────────────────────────────────────────────────
    if args.smoke:
        print("[smoke] tiny synthetic test of long-context FT path", flush=True)
        # Build a 2-layer 64-d model that fits anywhere, seq_len matched to
        # T_virtual so the RoPE buffer can index any position we sample.
        T_content_s = 32
        T_virtual_s = 256
        T_train_old_s = 32
        n_chunks_s = 2
        model = FullORN(
            d=64, n_q_heads=4, n_kv_heads=2, d_head=16,
            n_layers=2, vocab=200, seq_len=T_virtual_s + 16,
            ffn_width_mult=2,
        ).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[smoke] model params={n_params/1e6:.3f}M", flush=True)

        # NTK rescale check (using D=16 → exponent 16/14 = 1.143)
        theta_new = ntk_aware_theta(10000.0, T_train_old_s, T_virtual_s, 16)
        print(f"[smoke] NTK theta: 10000 → {theta_new:.0f}  "
              f"(expansion {T_virtual_s/T_train_old_s:.0f}×, "
              f"NTK exponent {16/14:.3f})", flush=True)
        rebuild_rope_freqs(model, T_virtual_s, theta_new, device)
        print(f"[smoke] new rope_freqs shape: {tuple(model.rope_freqs.shape)}",
              flush=True)

        # Synthetic PoSE-style sampler: random tokens + random positions
        # in [0, T_virtual_s), sampling with chunks for realism.
        def synth_batch(B):
            tokens = torch.randint(0, 200, (B, T_content_s + 1), device=device)
            # Two-chunk PoSE positions
            positions = torch.zeros(B, T_content_s, dtype=torch.long, device=device)
            for b in range(B):
                cs = T_content_s // 2
                u1 = np.random.randint(0, T_virtual_s - T_content_s)
                u2 = np.random.randint(u1 + cs, T_virtual_s - (T_content_s - cs))
                positions[b, :cs] = torch.arange(u1, u1 + cs, device=device)
                positions[b, cs:] = torch.arange(u2, u2 + (T_content_s - cs),
                                                  device=device)
            return tokens, positions

        opt = torch.optim.AdamW(model.parameters(), lr=1e-3,
                                  weight_decay=0.01)
        # Run 20 steps and confirm: (a) loss decreases, (b) no NaN, (c) the
        # forward path actually consumes positions (test: shuffling positions
        # changes the loss).
        losses = []
        for step in range(20):
            tokens, positions = synth_batch(B=2)
            x = tokens[:, :-1]; y = tokens[:, 1:]
            opt.zero_grad(set_to_none=True)
            _, loss_per_tok = L_forward_pose(model, x, positions, targets=y)
            loss = loss_per_tok.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        print(f"[smoke] step 0 loss={losses[0]:.4f}  "
              f"step 19 loss={losses[19]:.4f}  "
              f"decreased={losses[19] < losses[0]}", flush=True)
        assert not any(math.isnan(l) for l in losses), "NaN in smoke losses"

        # Test that positions actually matter: with same tokens, different
        # positions, output differs.
        tokens, pos1 = synth_batch(B=1)
        x = tokens[:, :-1]
        pos2 = torch.randperm(T_virtual_s, device=device)[:T_content_s].unsqueeze(0)
        with torch.no_grad():
            l1, _ = L_forward_pose(model, x, pos1[:1], targets=None)
            l2, _ = L_forward_pose(model, x, pos2, targets=None)
        diff = (l1 - l2).abs().max().item()
        print(f"[smoke] same-tokens-diff-positions logit diff = {diff:.4f}  "
              f"(should be > 0)", flush=True)
        assert diff > 1e-6, "positions don't affect output — RoPE wiring broken"

        print("[smoke] PASSED ✓", flush=True)
        return

    # ────────────────────────────────────────────────────────────────
    # Full path: load a real checkpoint and fine-tune.
    # ────────────────────────────────────────────────────────────────
    assert args.policy and args.data_dir and args.out, \
        "--policy, --data_dir, --out are required (or use --smoke)"

    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / (Path(args.out).stem + ".log")
    logf = open(log_path, "a", buffering=1)
    t0 = time.time()
    def log(ev):
        ev["t"] = round(time.time() - t0, 1)
        logf.write(json.dumps(ev, default=float) + "\n")
        print(json.dumps(ev, default=float), flush=True)

    log({"event": "start", "args": vars(args), "trainer": "long-context"})

    # Load checkpoint and rebuild the model with seq_len enlarged to
    # accommodate the new T_virtual. We deliberately CHANGE seq_len from
    # whatever was in the checkpoint config — the rope_freqs buffer is
    # non-persistent and will be regenerated.
    print(f"[load] {args.policy}", flush=True)
    ckpt = torch.load(args.policy, map_location=device, weights_only=False)
    cfg = ckpt["config"]; mc = dict(cfg["model"])
    mc["seq_len"] = args.T_virtual + 16  # Enlarge for long-T eval reach
    arch = cfg.get("arch") or "orn"
    assert arch == "orn", f"long-context FT only supports orn arch, got {arch}"

    model = build_orn(mc).to(device)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    log({"event": "model_loaded",
         "params_M": sum(p.numel() for p in model.parameters()) / 1e6,
         "missing_keys_n": len(missing),
         "unexpected_keys_n": len(unexpected),
         "seq_len_new": mc["seq_len"]})

    # NTK-aware RoPE rescaling. d_head from the model config.
    theta_old = 10000.0
    theta_new = ntk_aware_theta(theta_old, args.T_train_old, args.T_virtual,
                                  mc["d_head"])
    rebuild_rope_freqs(model, args.T_virtual, theta_new, device)
    log({"event": "rope_rescaled",
         "theta_old": theta_old, "theta_new": theta_new,
         "T_train_old": args.T_train_old, "T_virtual": args.T_virtual,
         "d_head": mc["d_head"]})

    # Data loader. Uses the existing PoSE loader — same as the small-model
    # trainer. Positions are SAMPLED in [0, T_virtual) and FLOW THROUGH
    # the trainer (not discarded as in the original v3 trainer bug).
    train_loader = PoSELongShardLoader(
        args.data_dir, args.T_content, args.T_virtual, args.batch,
        device=device, seed=args.seed, n_chunks=args.n_chunks,
    )

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay,
    )
    def lr_at(step):
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        prog = (step - args.warmup) / max(1, args.steps - args.warmup)
        prog = min(max(prog, 0.0), 1.0)
        return args.lr_min + 0.5 * (args.lr - args.lr_min) * (1 + np.cos(np.pi * prog))

    t_step = time.time()
    for step in range(args.steps):
        cur_lr = lr_at(step)
        for pg in opt.param_groups:
            pg["lr"] = cur_lr
        opt.zero_grad(set_to_none=True)
        acc_loss = 0.0
        for _ in range(args.grad_accum):
            tokens, positions = train_loader.get_batch()  # POSITIONS USED
            x = tokens[:, :-1]; y = tokens[:, 1:]
            x_pos = positions[:, :-1]
            with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                     enabled=(device == "cuda")):
                _, loss_per_tok = L_forward_pose(model, x, x_pos, targets=y)
                loss = loss_per_tok.mean()
            (loss / args.grad_accum).backward()
            acc_loss += loss.item()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        loss_v = acc_loss / args.grad_accum

        if (step + 1) % args.log_every == 0 or step == 0:
            dt = time.time() - t_step
            tok_per_step = args.batch * args.grad_accum * args.T_content
            tok_per_sec = tok_per_step * args.log_every / dt if step > 0 else 0
            log({
                "event": "step", "step": step + 1,
                "loss": round(loss_v, 4),
                "lr": round(cur_lr, 9),
                "gn": round(float(gn), 3),
                "tok_per_sec": int(tok_per_sec),
            })
            t_step = time.time()

        if (step + 1) % args.ckpt_every == 0 or (step + 1) == args.steps:
            torch.save({
                "model": model.state_dict(),
                "config": {**cfg, "model": mc,
                            "long_context": {"theta_new": theta_new,
                                              "T_virtual": args.T_virtual,
                                              "T_train_old": args.T_train_old}},
                "args": vars(args),
                "trainer": "long-context",
            }, args.out)
            log({"event": "ckpt", "step": step + 1, "path": args.out})

    log({"event": "done"})
    logf.close()


if __name__ == "__main__":
    main()
