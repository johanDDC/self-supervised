from dataclasses import dataclass
from typing import Tuple


@dataclass()
class GPTConfig():
    model_name: str = "t5-small"

    lr: float = 1e-4
    weight_decay: float = 1e-2
    betas: Tuple[float] = (0.9, 0.999)

    pct_start: float = 0.2
    anneal_strategy: str = "linear"
    div_factor: float = 1e2

    max_length:int = 512

    project_name: str = "self_supervised_nlp"
    entity_name: str = "johan_ddc_team"
    run_name: str = "T5_GPT_pretrain"
