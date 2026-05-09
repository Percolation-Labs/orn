"""MLX serving stack for ORN function-calling checkpoints.

A faithful port of the PyTorch FullORN architecture to Apple's [MLX](https://github.com/ml-explore/mlx)
framework, with KV cache, fp16 weights, and optional `mlx.nn.quantize`-based
int4/int8 quantization. The whole stack is self-contained — it loads the
same `.pt` checkpoint as the PyTorch backend but runs inference on MLX's
Metal kernels, which are typically 1.5-3× faster than PyTorch MPS for
this model class on Apple Silicon.

See `drafts/post_training_a_600m_orn.md` § "What the inference recipe looks
like" for the staged improvements that land at ~10s per tool call on M4
when MLX + int4 quantization are combined.

Public modules:
- `orn_mlx`        MLX port of FullORN (ORNV3) with KV-cache-aware forward
- `load`           PyTorch `.pt` -> MLX param tree translation
- `generate`       prefill + cached decode, token-by-token sampling
- `quantize`       `mlx.nn.quantize` wrapper with class predicate (FFN +
                   attention projections + embedding quantized; shared M
                   and small high-leverage params kept at fp16)
- `benchmark`      single-prompt timing benchmark, no server (direct backend)
- `eval_quality`   fp16 vs int4 quality comparison across the FC eval
"""

DEFAULT_FC_CHECKPOINT = "mr-saoirse/orn-v3-3b-fc-sft"
