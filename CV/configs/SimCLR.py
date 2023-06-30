from dataclasses import dataclass


@dataclass()
class SimCLRConfig:
    model_name: str = "SimCLR"

    lr: float = 3e-4
    weight_decay: float = 1e-4

    scheduler_anneal_strat: str = "cos"
    scheduler_num_warmup_epochs:int = 10

    loss_temperature:float = 0.05

    batch_size: int = 256
    num_epochs: int = 50

    project_name: str = "self_supervised_vision"
    entity_name: str = "TBD"
    run_name: str = "SimCLR"
