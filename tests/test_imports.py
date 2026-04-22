"""Smoke test: the package, submodules, and CLI import without error."""


def test_top_level_imports():
    import orn
    from orn import (
        ORN, ORNV2, ORNV3, CORN, LORN, Transformer,
        ORNConfig, ORNV3Config, LORNConfig,
    )
    assert orn.__version__


def test_submodules_import():
    import orn.cli            # noqa: F401
    import orn.training       # noqa: F401
    import orn.data           # noqa: F401
    import orn.diagnostics    # noqa: F401
    import orn.eval           # noqa: F401
    import orn.utils          # noqa: F401
    import orn.reproduce      # noqa: F401


def test_build_model_dispatch():
    from orn import build_model
    # Each arch gets a tiny but self-consistent config (d == n_q_heads * d_head
    # where required).
    configs = {
        "orn_v1": dict(d_model=64, n_layers=2, n_heads=4, d_corr=16,
                       vocab_size=200, seq_len=32),
        "orn_v2": dict(d_model=64, n_layers=2, n_q_heads=4, n_kv_heads=2,
                       d_head=16, d_corr=16, vocab_size=200, seq_len=32),
        "orn_v3": dict(d_model=64, n_layers=2, n_q_heads=4, n_kv_heads=2,
                       d_head=16, d_corr=16, vocab_size=200, seq_len=32,
                       ffn_width_mult=2),
        "corn":   dict(d_model=64, n_layers=2, n_heads=4, vocab_size=200, seq_len=32),
        "transformer": dict(d_model=64, n_layers=2, n_heads=4, vocab_size=200, seq_len=32),
    }
    for arch, cfg in configs.items():
        m = build_model(arch, cfg)
        assert sum(p.numel() for p in m.parameters()) > 0

    lorn = build_model("lorn_v4", dict(vocab_size=200, T_max=32, d_r=32, d_l=64,
                                       l_layers=2, l_q_heads=4, l_kv_heads=2, l_d_head=16))
    assert sum(p.numel() for p in lorn.parameters()) > 0
