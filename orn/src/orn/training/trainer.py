"""
ORN Training Engine
====================

Resumable, checkpointed training with:
- Cosine LR schedule with linear warmup
- M spectral diagnostics at every eval
- Text generation samples at regular intervals
- JSONL logging for all metrics
- Gradient clipping, weight decay, mixed precision (optional)

Designed for both local (M4, single GPU) and remote (Vast.ai) training.

Best practices for Vast.ai runs:
- Use gradient accumulation to simulate larger batch sizes
- Enable torch.compile() for 20-40% speedup on A100/4090/5090
- Set checkpoint_every_tokens to ~100M for long runs
- Use bf16 mixed precision on Ampere+ GPUs
- Monitor M rank — if it doesn't drop below d/2 by 500M tokens, something's wrong
"""

from __future__ import annotations
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


def _infer_arch(model) -> str:
    """Map the model class name back to its `arch` key used by build_model."""
    cls = type(model).__name__
    return {
        "ORN": "orn_v1",
        "ORNV2": "orn_v2",
        "ORNV3": "orn_v3",
        "CORN": "corn",
        "LORN": "lorn_v4",
        "Transformer": "transformer",
    }.get(cls, cls.lower())


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    # Optimiser
    lr: float = 2e-4
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0

    # Schedule
    warmup_steps: int = 1000
    max_tokens: int = 500_000_000  # 500M tokens
    batch_size: int = 32
    seq_len: int = 256  # overridden by model config

    # Gradient accumulation (effective_batch = batch_size * grad_accum_steps)
    grad_accum_steps: int = 1

    # Logging
    log_every_steps: int = 100
    eval_every_steps: int = 500
    generate_every_steps: int = 2000
    checkpoint_every_tokens: int = 100_000_000  # 100M tokens

    # Hardware
    compile: bool = False       # torch.compile (20-40% speedup on Ampere+)
    mixed_precision: bool = False  # bf16 on CUDA, otherwise disabled
    num_eval_batches: int = 30

    # Generation
    generate_prompts: list[str] = field(default_factory=lambda: [
        "The research shows that",
        "Once upon a time",
    ])

    # Output
    output_dir: str = "output"

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d: dict) -> "TrainingConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class Trainer:
    """
    Training engine for ORN/CORN/Transformer models.

    Handles:
    - Training loop with warmup + cosine LR
    - Periodic evaluation, generation, checkpointing
    - M spectral diagnostics at every eval (for ORN/CORN)
    - JSONL logging
    - Resume from checkpoint

    Usage:
        from orn import ORNV2, ORNConfig
        from orn.training import Trainer, TrainingConfig
        from orn.data import ShardedDataLoader

        model = ORNV2(ORNConfig(d_model=512, n_layers=24, ...))
        train_loader = ShardedDataLoader("data/", 256, 32, "cuda")
        val_loader = ShardedDataLoader("data/", 256, 32, "cuda", eval_mode=True)
        trainer = Trainer(model, TrainingConfig(), train_loader, val_loader)
        trainer.train()
    """

    def __init__(self, model, config: TrainingConfig,
                 train_loader, val_loader=None,
                 device: Optional[torch.device] = None,
                 resume_from: Optional[str] = None,
                 loss_fn=None):
        from orn.training.losses import get_loss_fn
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn if loss_fn is not None else get_loss_fn("lm_ce")

        # Device
        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.model = self.model.to(self.device)

        # torch.compile
        if config.compile and hasattr(torch, "compile"):
            print("  Compiling model with torch.compile()...")
            self.model = torch.compile(self.model)

        # Output dirs
        self.output_dir = Path(config.output_dir)
        self.ckpt_dir = self.output_dir / "checkpoints"
        self.log_dir = self.output_dir / "logs"
        self.plot_dir = self.output_dir / "plots"
        for d in [self.ckpt_dir, self.log_dir, self.plot_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Optimiser (separate param groups for weight decay)
        decay_params = []
        no_decay_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if param.dim() >= 2:
                    decay_params.append(param)
                else:
                    no_decay_params.append(param)

        self.optimizer = torch.optim.AdamW([
            {"params": decay_params, "weight_decay": config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ], lr=config.lr, betas=config.betas)

        # LR schedule
        self.total_steps = config.max_tokens // (
            config.batch_size * config.seq_len * config.grad_accum_steps
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, self._lr_fn
        )

        # Mixed precision
        self.scaler = None
        if config.mixed_precision and self.device.type == "cuda":
            self.scaler = torch.amp.GradScaler("cuda")

        # Tokenizer for generation
        self._enc = None

        # State
        self.step = 0
        self.tokens_seen = 0
        self.best_val = float("inf")

        # Resume
        if resume_from:
            self._resume(resume_from)

    def _lr_fn(self, step: int) -> float:
        """Warmup then cosine decay."""
        warmup = self.config.warmup_steps
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / (self.total_steps - warmup + 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    @property
    def enc(self):
        if self._enc is None:
            import tiktoken
            self._enc = tiktoken.get_encoding("gpt2")
        return self._enc

    def train(self):
        """Run the full training loop."""
        tc = self.config
        tokens_per_step = tc.batch_size * tc.seq_len * tc.grad_accum_steps

        self._print_header()
        log_file = open(self.log_dir / "training_log.jsonl", "a")
        t0 = time.time()
        val_loss = None

        for step in range(self.step, self.total_steps):
            self.step = step
            self.model.train()

            # Gradient accumulation
            total_loss = 0.0
            self.optimizer.zero_grad()

            for micro_step in range(tc.grad_accum_steps):
                x, y = self.train_loader.get_batch()

                if self.scaler and self.device.type == "cuda":
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        loss, _ = self.loss_fn(self.model, x, y)
                    loss = loss / tc.grad_accum_steps
                    self.scaler.scale(loss).backward()
                else:
                    loss, _ = self.loss_fn(self.model, x, y)
                    loss = loss / tc.grad_accum_steps
                    loss.backward()

                total_loss += loss.item()

            if self.scaler:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), tc.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), tc.grad_clip)
                self.optimizer.step()

            self.scheduler.step()
            self.tokens_seen += tokens_per_step

            # ── Logging ──
            if (step + 1) % tc.log_every_steps == 0:
                elapsed = time.time() - t0
                tps = self.tokens_seen / (elapsed + 1e-6)
                lr = self.scheduler.get_last_lr()[0]
                print(
                    f"  step {step+1:>7d} | loss {total_loss:.4f} | "
                    f"lr {lr:.6f} | {self.tokens_seen:,} tok | "
                    f"{tps:.0f} tok/s",
                    flush=True,
                )

            # ── Eval ──
            if (step + 1) % tc.eval_every_steps == 0:
                val_loss = self._evaluate()
                diag = self._spectral_diag()

                diag_str = ""
                if diag:
                    diag_str = (
                        f" | M rank={diag.get('eff_rank_90', '?')} "
                        f"κ={diag.get('condition_number', 0):.0f} "
                        f"asym={diag.get('asymmetry', 0):.3f}"
                    )
                print(f"  {'':>8s}   val {val_loss:.4f}{diag_str}", flush=True)

                entry = {
                    "step": step + 1,
                    "tokens": self.tokens_seen,
                    "train_loss": total_loss,
                    "val_loss": val_loss,
                    "lr": self.scheduler.get_last_lr()[0],
                    "wall_time": time.time() - t0,
                }
                if diag:
                    entry.update({
                        "M_rank": diag.get("eff_rank_90"),
                        "M_cond": diag.get("condition_number"),
                        "M_asym": diag.get("asymmetry"),
                    })
                log_file.write(json.dumps(entry) + "\n")
                log_file.flush()

                if val_loss < self.best_val:
                    self.best_val = val_loss

            # ── Generate samples ──
            if (step + 1) % tc.generate_every_steps == 0:
                self._generate_samples()

            # ── Checkpoint ──
            if (
                self.tokens_seen > 0
                and self.tokens_seen % tc.checkpoint_every_tokens < tokens_per_step
            ):
                self._checkpoint(val_loss)

            if self.tokens_seen >= tc.max_tokens:
                break

        # Final checkpoint and report
        self._checkpoint(val_loss, tag="final")
        self._generate_report()

        elapsed = time.time() - t0
        print(f"\n  Training complete.", flush=True)
        print(f"  Steps: {self.step + 1:,}", flush=True)
        print(f"  Tokens: {self.tokens_seen:,}", flush=True)
        print(f"  Time: {elapsed:.0f}s ({elapsed/3600:.1f}h)", flush=True)
        print(f"  Best val: {self.best_val:.4f}", flush=True)

        log_file.close()

    def _evaluate(self) -> float:
        """Run evaluation and return mean val loss."""
        if self.val_loader is None:
            return float("inf")

        self.model.eval()
        losses = []
        with torch.no_grad():
            for _ in range(self.config.num_eval_batches):
                x, y = self.val_loader.get_batch()
                loss, _ = self.loss_fn(self.model, x, y)
                losses.append(loss.item())
        return float(np.mean(losses))

    def _spectral_diag(self) -> dict:
        """Extract M spectral diagnostics if model supports it."""
        if hasattr(self.model, "spectral_diagnostics"):
            return self.model.spectral_diagnostics()
        # Handle compiled models
        if hasattr(self.model, "_orig_mod") and hasattr(self.model._orig_mod, "spectral_diagnostics"):
            return self.model._orig_mod.spectral_diagnostics()
        return {}

    def _generate_samples(self):
        """Generate text samples from prompts."""
        model = self.model
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod

        if not hasattr(model, "generate"):
            return

        model.eval()
        for prompt in self.config.generate_prompts:
            try:
                ids = torch.tensor(self.enc.encode(prompt), device=self.device)
                out = model.generate(ids, max_new=80)
                text = self.enc.decode(out.tolist())
                print(f"  [{prompt}] {text[:200]}", flush=True)
            except Exception as e:
                print(f"  [gen error: {e}]", flush=True)

    def _checkpoint(self, val_loss: Optional[float] = None, tag: Optional[str] = None):
        """Save a checkpoint."""
        model = self.model
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod

        ckpt = {
            "model": model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "step": self.step + 1,
            "tokens_seen": self.tokens_seen,
            "val_loss": val_loss,
            "config": self.config.to_dict(),
            "spectral": self._spectral_diag(),
        }
        if hasattr(model, "config"):
            mc = model.config
            mc_dict = mc.to_dict() if hasattr(mc, "to_dict") else dict(mc)
            mc_dict.setdefault("arch", _infer_arch(model))
            ckpt["model_config"] = mc_dict
        elif hasattr(model, "cfg"):   # LORN uses .cfg, not .config
            mc = model.cfg
            mc_dict = mc.to_dict() if hasattr(mc, "to_dict") else dict(vars(mc))
            mc_dict.setdefault("arch", _infer_arch(model))
            ckpt["model_config"] = mc_dict

        if tag:
            path = self.ckpt_dir / f"{tag}.pt"
        else:
            path = self.ckpt_dir / f"orn_{self.tokens_seen // 1_000_000}M_tok.pt"

        torch.save(ckpt, path)
        torch.save(ckpt, self.ckpt_dir / "latest.pt")
        print(f"  Checkpoint: {path} ({self.tokens_seen:,} tokens)", flush=True)

    def _resume(self, path: str):
        """Resume training from a checkpoint."""
        print(f"  Resuming from {path}...", flush=True)
        ckpt = torch.load(path, map_location=self.device)

        model = self.model
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod
        model.load_state_dict(ckpt["model"])

        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.step = ckpt["step"]
        self.tokens_seen = ckpt["tokens_seen"]
        print(f"  Resumed at step {self.step}, {self.tokens_seen:,} tokens", flush=True)

    def _generate_report(self):
        """Generate training report plot from log file."""
        log_path = self.log_dir / "training_log.jsonl"
        if not log_path.exists():
            return

        entries = []
        with open(log_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))

        if not entries:
            return

        try:
            from orn.diagnostics.plots import plot_training_report
            plot_training_report(
                entries,
                model_name=type(self.model).__name__,
                save_path=self.plot_dir / "training_report.png",
            )
            print(f"  Report: {self.plot_dir / 'training_report.png'}", flush=True)
        except Exception as e:
            print(f"  Report generation failed: {e}", flush=True)

    def _print_header(self):
        """Print training header."""
        tc = self.config
        tokens_per_step = tc.batch_size * tc.seq_len * tc.grad_accum_steps
        model_name = type(self.model).__name__
        if hasattr(self.model, "_orig_mod"):
            model_name = type(self.model._orig_mod).__name__

        param_info = ""
        model = self.model._orig_mod if hasattr(self.model, "_orig_mod") else self.model
        if hasattr(model, "count_params"):
            params = model.count_params()
            param_info = f"  Parameters: {params['total']:,} (coupling: {params.get('coupling_pct', 0):.1f}%)"

        print(f"╔══════════════════════════════════════════════════════════╗", flush=True)
        print(f"║  {model_name} Training", flush=True)
        print(f"╚══════════════════════════════════════════════════════════╝", flush=True)
        print(f"  Device: {self.device}", flush=True)
        print(f"  Steps: {self.total_steps:,}, Tokens: {tc.max_tokens:,}", flush=True)
        print(f"  Batch: {tc.batch_size}×{tc.seq_len}×{tc.grad_accum_steps} = {tokens_per_step:,} tok/step", flush=True)
        if param_info:
            print(param_info, flush=True)
        if tc.compile:
            print(f"  torch.compile: enabled", flush=True)
        if tc.mixed_precision:
            print(f"  Mixed precision: bf16", flush=True)
        print(flush=True)
