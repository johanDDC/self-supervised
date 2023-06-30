from dataclasses import dataclass
from typing import Union, Tuple, List


@dataclass()
class BYOLConfig:
    model_name: str = "SimCLR"

    lr: float = 3e-4
    weight_decay: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.999)

    scheduler_anneal_strat: str = "cos"
    scheduler_num_warmup_epoches:int = 10

    EMA:float = 0.996

    batch_size: int = 256
    num_epochs: int = 50

    project_name: str = "self_supervised_vision"
    entity_name: str = "TBD"
    run_name: str = "SimCLR"
