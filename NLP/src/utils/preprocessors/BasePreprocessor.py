from transformers import T5TokenizerFast


class BasePreprocessor:
    def __init__(self, tokenizer: T5TokenizerFast):
        self.tokenizer = tokenizer

    def __call__(self, sample):
        pass
