from dataclasses import dataclass

from NLP.configs.supervised.SummarizationConfig import SummarizationConfig

_no_default = "<NO DEFAULT>"
_no_default_float = float("inf")


@dataclass()
class TuneSummarization(SummarizationConfig):
    pretrain_model_path: str = _no_default
    # ratio of steps before model fully unfreezed
    model_warmup:float = _no_default_float

    lr: float = 6.5e-5
    anneal_strategy: str = "cos"
    run_name: str = "summarization_ft"

    def __post_init__(self):
        if self.pretrain_model_path == _no_default:
            raise TypeError("__init__ missing 1 required argument: 'pretrain_model_path'")
        if self.model_warmup == _no_default_float:
            raise TypeError("__init__ missing 1 required argument: 'model_warmup'")
