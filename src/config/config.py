from dataclasses import dataclass

from hydra.core.config_store import ConfigStore


@dataclass
class TrainConfig:
    random_seed: int = 42
    pretrained: str = "distilbert-base-uncased"
    npratio: int = 4
    history_size: int = 50
    batch_size: int = 16
    gradient_accumulation_steps: int = 8  # batch_size = 16 x 8 = 128
    epochs: int = 3
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    max_len: int = 30


cs = ConfigStore.instance()

cs.store(name="train_config", node=TrainConfig)
