from dataclasses import dataclass
from typing import Tuple, Union

from NLP.configs.BaseConfig import BaseConfig


@dataclass()
class SummarizationConfig(BaseConfig):
    model_name: str = "t5-small"
    dataset_name: str = "xsum"

    lr: float = 1e-4
    weight_decay: float = 1e-2
    betas: Tuple[float] = (0.9, 0.999)

    pct_start: float = 0.2
    anneal_strategy: str = "linear"
    div_factor: float = 1e2

    project_name: str = "self_supervised_nlp"
    entity_name: str = "johan_ddc_team"
    run_name: str = "t5_summarization_supervised"

    log_keys: Tuple[str] = ("ROUGE",)
    log_ids: Tuple[str] = ("rouge1",)
    key_metric: str = "rouge1"

    dataset_info: Union[None, str] = None
    dataset_split: Union[None, int] = 50

    max_input_length: int = 512
    max_target_length: int = 128
