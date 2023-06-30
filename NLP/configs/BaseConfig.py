from dataclasses import dataclass
from typing import Union, Tuple


@dataclass()
class BaseConfig:
    model_name: str
    dataset_name: str

    lr: float
    weight_decay: float
    betas: Tuple[float]

    pct_start: float
    anneal_strategy: str
    div_factor: float

    project_name: str
    entity_name: str
    run_name: str

    log_keys: Tuple[str]
    log_ids: Tuple[str]
    key_metric: str

    dataset_info: Union[None, str] = None
    dataset_split: Union[None, int] = None

