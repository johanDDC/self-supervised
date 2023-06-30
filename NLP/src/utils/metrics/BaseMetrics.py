from transformers import T5TokenizerFast


class BaseMetrics:
    def __init__(self, tokenizer: T5TokenizerFast):
        self.tokenizer = tokenizer

    def __call__(self, eval_preds):
        pass