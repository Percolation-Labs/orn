"""Training engine + loss adapters + JSON config loader."""
from orn.training.trainer import Trainer, TrainingConfig
from orn.training.losses import get_loss_fn
from orn.training.config_loader import load_config, list_configs, CONFIG_DIR

__all__ = ["Trainer", "TrainingConfig", "get_loss_fn",
           "load_config", "list_configs", "CONFIG_DIR"]
