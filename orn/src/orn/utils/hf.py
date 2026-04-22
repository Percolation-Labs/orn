"""HuggingFace helpers: tokenizer pull, checkpoint push/pull.

Requires `pip install -e '.[data]'`. Token read from HF_TOKEN (via .env or shell).
"""
from __future__ import annotations

from pathlib import Path

from orn.utils.env import get_env


def hf_token() -> str | None:
    return get_env("HF_TOKEN")


def get_smollm2_tokenizer():
    """Return the SmolLM2 tokenizer (vocab 49152). Cached under HF_HOME."""
    try:
        from transformers import AutoTokenizer
    except ImportError as e:
        raise ImportError("pip install transformers (or `pip install -e '.[data]'`)") from e
    return AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M", token=hf_token())


def get_tiktoken_gpt2():
    """Return the GPT-2 tiktoken tokenizer (vocab 50257). Local, no HF pull."""
    import tiktoken
    return tiktoken.get_encoding("gpt2")


def get_tokenizer(name: str = "tiktoken-gpt2"):
    """name: 'tiktoken-gpt2' (default) or 'smollm2'."""
    name = name.lower().replace("_", "-")
    if name == "tiktoken-gpt2":
        return get_tiktoken_gpt2()
    if name == "smollm2":
        return get_smollm2_tokenizer()
    raise ValueError(f"Unknown tokenizer '{name}'. Options: tiktoken-gpt2, smollm2")


def push_checkpoint(local_path: str | Path, repo_id: str, private: bool = True) -> str:
    """Upload a checkpoint .pt to HuggingFace. Returns the uploaded URL."""
    from huggingface_hub import HfApi, create_repo
    api = HfApi(token=hf_token())
    create_repo(repo_id, token=hf_token(), private=private, exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=Path(local_path).name,
        repo_id=repo_id,
    )
    return f"https://huggingface.co/{repo_id}/blob/main/{Path(local_path).name}"


def pull_model(model_name: str, cache_dir: str | Path | None = None):
    """Pull a baseline model for eval (gpt2, pythia-70m, pythia-410m, smollm2-135m)."""
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as e:
        raise ImportError("pip install transformers") from e
    known = {
        "gpt2": "gpt2",
        "smollm2-135m": "HuggingFaceTB/SmolLM2-135M",
        "pythia-70m": "EleutherAI/pythia-70m",
        "pythia-410m": "EleutherAI/pythia-410m",
    }
    repo = known.get(model_name.lower(), model_name)
    tok = AutoTokenizer.from_pretrained(repo, cache_dir=cache_dir, token=hf_token())
    mdl = AutoModel.from_pretrained(repo, cache_dir=cache_dir, token=hf_token())
    return mdl, tok
