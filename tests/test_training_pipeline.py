"""Smoke test: trainer runs a handful of steps on synthetic data for each loss type."""
import torch
import pytest

from orn.models.orn import ORN, ORNConfig
from orn.models.lorn import lorn_v4_smoke
from orn.training import Trainer, TrainingConfig, get_loss_fn


class _Loader:
    """Synthetic batch producer that looks like a ShardedDataLoader."""
    def __init__(self, vocab, seq_len, batch_size, device):
        self.v, self.s, self.b, self.d = vocab, seq_len, batch_size, device

    def get_batch(self):
        x = torch.randint(0, self.v, (self.b, self.s), device=self.d)
        y = torch.roll(x, -1, dims=1)
        return x, y


def test_trainer_lm_ce(tmp_path):
    torch.manual_seed(0)
    model = ORN(ORNConfig(d_model=32, n_layers=2, n_heads=4, d_corr=8,
                          vocab_size=64, seq_len=16, version=1))
    tc = TrainingConfig(
        lr=1e-3, warmup_steps=2, max_tokens=64 * 4,   # 4 micro-batches
        batch_size=4, seq_len=16, grad_accum_steps=1,
        log_every_steps=100, eval_every_steps=100,
        generate_every_steps=100, checkpoint_every_tokens=10_000_000,
        num_eval_batches=2, output_dir=str(tmp_path),
        compile=False, mixed_precision=False,
    )
    loader = _Loader(64, 16, 4, "cpu")
    trainer = Trainer(model, tc, loader, loader, device=torch.device("cpu"))
    trainer.train()
    assert (tmp_path / "checkpoints" / "final.pt").exists()


def test_trainer_lorn_vq(tmp_path):
    torch.manual_seed(0)
    model = lorn_v4_smoke(vocab_size=64, T_max=16)
    tc = TrainingConfig(
        lr=1e-3, warmup_steps=2, max_tokens=16 * 4 * 2,
        batch_size=4, seq_len=16, grad_accum_steps=1,
        log_every_steps=100, eval_every_steps=100,
        generate_every_steps=100, checkpoint_every_tokens=10_000_000,
        num_eval_batches=2, output_dir=str(tmp_path),
        compile=False, mixed_precision=False,
    )
    loader = _Loader(64, 16, 4, "cpu")
    trainer = Trainer(model, tc, loader, loader, device=torch.device("cpu"),
                      loss_fn=get_loss_fn("lm_ce+lorn_vq"))
    trainer.train()
    assert (tmp_path / "checkpoints" / "final.pt").exists()
