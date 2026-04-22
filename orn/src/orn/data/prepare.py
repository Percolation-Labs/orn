"""
Data preparation: download and tokenise datasets.
====================================================

Prepares tokenised shards for training. Supports:
- TinyStories (small, fast, good for development)
- OpenWebText (standard LM benchmark)
- FineWeb-Edu (high quality web text, used for V1)

Output: shard_XXXX.pt files in the target directory, each containing
a 1D tensor of token IDs (~100M tokens per shard).
"""

from __future__ import annotations
import torch
from pathlib import Path
from typing import Optional


def prepare_tinystories(
    output_dir: str | Path = "data",
    max_tokens: int = 50_000_000,
    shard_size: int = 10_000_000,
    encoding: str = "gpt2",
):
    """
    Download and tokenise TinyStories.

    Args:
        output_dir: Directory for output shards
        max_tokens: Maximum tokens to process
        shard_size: Tokens per shard file
        encoding: Tiktoken encoding name
    """
    from datasets import load_dataset
    import tiktoken

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    enc = tiktoken.get_encoding(encoding)
    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)

    tokens = []
    shard_idx = 0
    total = 0

    print(f"Tokenising TinyStories → {output_dir}/")
    for item in ds:
        toks = enc.encode(item["text"])
        tokens.extend(toks)
        total += len(toks)

        while len(tokens) >= shard_size:
            shard = torch.tensor(tokens[:shard_size], dtype=torch.int32)
            path = output_dir / f"shard_{shard_idx:04d}.pt"
            torch.save(shard, path)
            print(f"  Shard {shard_idx}: {shard_size:,} tokens → {path}")
            tokens = tokens[shard_size:]
            shard_idx += 1

        if total >= max_tokens:
            break

    # Save remaining
    if tokens:
        shard = torch.tensor(tokens, dtype=torch.int32)
        path = output_dir / f"shard_{shard_idx:04d}.pt"
        torch.save(shard, path)
        print(f"  Shard {shard_idx}: {len(tokens):,} tokens → {path}")

    # Save a small validation set from the last shard
    val_ds = load_dataset("roneneldan/TinyStories", split="validation", streaming=True)
    val_tokens = []
    for item in val_ds:
        val_tokens.extend(enc.encode(item["text"]))
        if len(val_tokens) >= 500_000:
            break
    val_tensor = torch.tensor(val_tokens[:500_000], dtype=torch.int32)
    torch.save(val_tensor, output_dir / "val_tokens.pt")

    print(f"  Total: {total:,} tokens in {shard_idx + 1} shards")


def prepare_openwebtext(
    output_dir: str | Path = "data",
    max_tokens: int = 500_000_000,
    shard_size: int = 100_000_000,
    encoding: str = "gpt2",
):
    """
    Download and tokenise OpenWebText.

    Args:
        output_dir: Directory for output shards
        max_tokens: Maximum tokens to process
        shard_size: Tokens per shard file
        encoding: Tiktoken encoding name
    """
    from datasets import load_dataset
    import tiktoken

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    enc = tiktoken.get_encoding(encoding)
    ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)

    tokens = []
    shard_idx = 0
    total = 0

    print(f"Tokenising OpenWebText → {output_dir}/")
    for item in ds:
        toks = enc.encode(item["text"])
        tokens.extend(toks)
        total += len(toks)

        while len(tokens) >= shard_size:
            shard = torch.tensor(tokens[:shard_size], dtype=torch.int32)
            path = output_dir / f"shard_{shard_idx:04d}.pt"
            torch.save(shard, path)
            print(f"  Shard {shard_idx}: {shard_size:,} tokens → {path}")
            tokens = tokens[shard_size:]
            shard_idx += 1

        if total >= max_tokens:
            break

    if tokens:
        shard = torch.tensor(tokens, dtype=torch.int32)
        path = output_dir / f"shard_{shard_idx:04d}.pt"
        torch.save(shard, path)

    print(f"  Total: {total:,} tokens in {shard_idx + 1} shards")
