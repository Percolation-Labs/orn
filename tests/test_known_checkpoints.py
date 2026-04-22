"""Known-checkpoint registry tests.

The download paths are network-dependent and skipped if huggingface_hub is
missing or HF is unreachable. The registry + resolver are exercised offline.
"""
import pytest


def test_registry_listing():
    from orn.utils.checkpoints import known_checkpoints, get_spec
    names = [s.name for s in known_checkpoints()]
    assert "orn-v3-605m" in names
    assert "orn-v2-108m" in names
    for name in names:
        s = get_spec(name)
        assert s.arch in ("orn_v1", "orn_v2", "orn_v3", "corn", "lorn_v4", "transformer")
        assert s.repo_id.count("/") == 1
        assert s.filename.endswith(".pt")


def test_resolve_local_accepts_known_name_without_downloading(monkeypatch, tmp_path):
    """resolve_local returns (path, spec) when passed a known name, via fetch_checkpoint.

    We stub fetch so we don't hit the network.
    """
    from orn.utils import checkpoints as C

    fake = tmp_path / "fake.pt"
    fake.write_bytes(b"stub")
    monkeypatch.setattr(C, "fetch_checkpoint", lambda name, cache_dir=None: fake)

    path, spec = C.resolve_local("orn-v2-108m")
    assert path == fake
    assert spec is not None
    assert spec.name == "orn-v2-108m"


def test_resolve_local_rejects_bogus_names(tmp_path):
    from orn.utils.checkpoints import resolve_local
    with pytest.raises(FileNotFoundError):
        resolve_local("not-a-known-checkpoint-and-not-a-file")


def test_legacy_v2_remapper():
    """Keys in the old flat layout translate to the current wrapped layout."""
    from orn.utils.checkpoints import _remap_legacy_v2_keys
    import torch
    legacy = {
        "A": torch.zeros(1),
        "B": torch.zeros(1),
        "blocks.0.A": torch.zeros(1),
        "blocks.0.ln_corr.weight": torch.zeros(1),
        "blocks.0.gate.weight": torch.zeros(1),
        "blocks.0.gate.bias": torch.zeros(1),
        "blocks.0.correction.0.weight": torch.zeros(1),
        "blocks.0.correction.2.weight": torch.zeros(1),
    }
    out = _remap_legacy_v2_keys(legacy)
    assert "blocks.0.correction.norm.weight" in out
    assert "blocks.0.correction.gate.weight" in out
    assert "blocks.0.correction.gate.bias" in out
    assert "blocks.0.correction.correction.0.weight" in out
    # Top-level A/B preserved; per-block A preserved (shared-param re-registration).
    assert "A" in out and "blocks.0.A" in out
