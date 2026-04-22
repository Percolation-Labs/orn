"""
Data loading for ORN training.
===============================

ShardedDataLoader: Loads tokenised text from pre-sharded .pt files on disk.
Supports random batching, shard rotation, and resuming from a position.

Tokenisation uses tiktoken (GPT-2 BPE by default, 50257 tokens).
For V2 models targeting SmolLM compatibility, use vocab_size=49152.
"""

from __future__ import annotations
import torch
from pathlib import Path
from typing import Optional


def get_tokenizer(encoding: str = "gpt2"):
    """
    Get a tiktoken tokenizer.

    Args:
        encoding: Tiktoken encoding name. "gpt2" (50257 tokens) is standard.

    Returns:
        tiktoken.Encoding object with .encode() and .decode() methods.
    """
    import tiktoken
    return tiktoken.get_encoding(encoding)


class ShardedDataLoader:
    """
    Load tokenised text from pre-sharded .pt files for training.

    Data format: each shard is a 1D int32/int64 tensor of token IDs saved
    with torch.save(). Shards are named shard_0000.pt, shard_0001.pt, etc.

    Also supports a single tokens file (e.g., owt_tokens.pt) containing a
    dict with "train" and "val" keys.

    Usage:
        loader = ShardedDataLoader("data/", seq_len=256, batch_size=32, device="cuda")
        x, y = loader.get_batch()  # x: (B, S), y: (B, S) shifted by 1
    """

    def __init__(self, data_dir: str | Path, seq_len: int, batch_size: int,
                 device: torch.device | str = "cpu", eval_mode: bool = False):
        self.data_dir = Path(data_dir)
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.device = torch.device(device) if isinstance(device, str) else device

        # Find shards
        self.shard_paths = sorted(self.data_dir.glob("shard_*.pt"))
        if not self.shard_paths:
            # Fallback: single tokens file
            single = self.data_dir / "owt_tokens.pt"
            if not single.exists():
                single = self.data_dir / "tokens.pt"
            if single.exists():
                d = torch.load(single, weights_only=True)
                self.data = d["val" if eval_mode else "train"]
                self.shard_paths = None
            else:
                raise FileNotFoundError(
                    f"No shards (shard_*.pt) or token files in {data_dir}. "
                    f"Run `orn prepare-data` first."
                )
        else:
            self.current_shard_idx = 0
            self.data = torch.load(self.shard_paths[0], weights_only=True)

        self.pos = 0

    def get_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample a random batch of (input, target) pairs.

        Returns:
            x: (batch_size, seq_len) input token IDs
            y: (batch_size, seq_len) target token IDs (shifted by 1)
        """
        B, S = self.batch_size, self.seq_len

        # Advance shard if needed
        if self.pos + B * S + 1 >= len(self.data):
            if self.shard_paths and len(self.shard_paths) > 1:
                self.current_shard_idx = (self.current_shard_idx + 1) % len(self.shard_paths)
                self.data = torch.load(
                    self.shard_paths[self.current_shard_idx], weights_only=True
                )
                self.pos = 0
            else:
                self.pos = 0

        ix = torch.randint(len(self.data) - S - 1, (B,))
        x = torch.stack([self.data[i : i + S] for i in ix]).long().to(self.device)
        y = torch.stack([self.data[i + 1 : i + S + 1] for i in ix]).long().to(self.device)
        self.pos += B * S
        return x, y

    def __len__(self) -> int:
        """Approximate number of tokens in current shard."""
        return len(self.data)
