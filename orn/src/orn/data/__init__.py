"""
Data loading utilities.

- loaders.py: ShardedDataLoader for tokenised text, batching utilities
- synthetic.py: Synthetic tasks for controlled experiments (color matching, patterns)
- prepare.py: Download and tokenise datasets (TinyStories, OpenWebText, FineWeb-Edu)
"""

from __future__ import annotations
from orn.data.loaders import ShardedDataLoader, get_tokenizer
from orn.data.synthetic import make_color_matching_batch, make_pattern_batch
