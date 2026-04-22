"""Smoke test: every architecture builds, does a forward pass, and backprops."""
import torch

from orn.models.orn import ORN, ORNV2, ORNConfig
from orn.models.orn_v3 import ORNV3, ORNV3Config
from orn.models.corn import CORN, CORNConfig
from orn.models.lorn import lorn_v4_smoke
from orn.models.transformer import Transformer, TransformerConfig


def _run_lm(model, x, y):
    _, loss = model(x, targets=y)
    loss.backward()
    return loss.item()


def test_orn_v1():
    torch.manual_seed(0)
    cfg = ORNConfig(d_model=64, n_layers=2, n_heads=4, d_corr=16,
                    vocab_size=200, seq_len=32, version=1)
    m = ORN(cfg)
    x = torch.randint(0, 200, (2, 32))
    loss = _run_lm(m, x, x)
    assert loss > 0
    diag = m.spectral_diagnostics()
    assert "eff_rank_90" in diag


def test_orn_v2():
    torch.manual_seed(0)
    cfg = ORNConfig(d_model=64, n_layers=2, n_q_heads=4, n_kv_heads=2, d_head=16,
                    d_corr=16, vocab_size=200, seq_len=32, version=2)
    m = ORNV2(cfg)
    x = torch.randint(0, 200, (2, 32))
    loss = _run_lm(m, x, x)
    assert loss > 0


def test_orn_v3():
    torch.manual_seed(0)
    cfg = ORNV3Config(d_model=64, n_layers=2, n_q_heads=4, n_kv_heads=2, d_head=16,
                      d_corr=16, vocab_size=200, seq_len=32, ffn_width_mult=2)
    m = ORNV3(cfg)
    x = torch.randint(0, 200, (2, 32))
    loss = _run_lm(m, x, x)
    assert loss > 0
    # freeze / unfreeze
    m.freeze_backbone()
    assert not m.A.requires_grad
    m.unfreeze_backbone()
    assert m.A.requires_grad


def test_transformer():
    torch.manual_seed(0)
    cfg = TransformerConfig.from_dict(dict(d_model=64, n_layers=2, n_heads=4,
                                            vocab_size=200, seq_len=32))
    m = Transformer(cfg)
    x = torch.randint(0, 200, (2, 32))
    loss = _run_lm(m, x, x)
    assert loss > 0


def test_corn():
    torch.manual_seed(0)
    cfg = CORNConfig.from_dict(dict(d_model=64, n_layers=2, n_heads=4,
                                     vocab_size=200, seq_len=32))
    m = CORN(cfg)
    x = torch.randint(0, 200, (2, 32))
    loss = _run_lm(m, x, x)
    assert loss > 0


def test_lorn_joint_loss():
    torch.manual_seed(0)
    m = lorn_v4_smoke(vocab_size=200, T_max=32)
    x = torch.randint(0, 200, (2, 32))
    _, task_loss, aux = m(x, targets=x)
    joint = m.joint_loss(task_loss, aux)
    joint.backward()
    assert joint.item() > 0
    assert "chosen" in aux
