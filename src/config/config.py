from dataclasses import dataclass
from hydra.core.config_store import ConfigStore


@dataclass
class TrainConfig:
    pretrained: str = ("distilbert-base-uncased",)
    npratio: int = (4,)
    history_size: int = (30,)
    batch_size: int = (32,)
    epochs: int = (3,)
    learning_rate: float = (1e-4,)
    weight_decay: float = (0.0,)
    max_len: int = (30,)


cs = ConfigStore.instance()

cs.store(name="train_config", node=TrainConfig)
