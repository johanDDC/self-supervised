from dataclasses import dataclass
from typing import Union, Tuple


@dataclass()
class SupervisedConfig:
    model_name: str = "ResNet-18"

    lr: float = 1e-3
    weight_decay: float = 5e-2

    scheduler_milestones: Tuple[int, int] = (30, 40)

    mixup_alpha:float = 0.8

    batch_size: int = 64
    num_epochs: int = 50

    project_name: str = "self_supervised_vision"
    entity_name: str = "TBD"
    run_name: str = "supervised"
