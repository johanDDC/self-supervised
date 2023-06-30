from dataclasses import dataclass

from NLP.configs.supervised.QAConfig import QAConfig

_no_default_str = "<NO DEFAULT>"
_no_default_float = float("inf")


@dataclass()
class TuneQA(QAConfig):
    pretrain_model_path: str = _no_default_str
    # ratio of steps before model fully unfreezed
    model_warmup:float = _no_default_float

    lr:float = 2e-5
    run_name: str = "qa_ft"

    def __post_init__(self):
        if self.pretrain_model_path == _no_default_str:
            raise TypeError("__init__ missing 1 required argument: 'pretrain_model_path'")
        if self.model_warmup == _no_default_float:
            raise TypeError("__init__ missing 1 required argument: 'model_warmup'")
