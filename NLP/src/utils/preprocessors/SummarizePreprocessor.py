from transformers import T5TokenizerFast

from NLP.src.utils.preprocessors.BasePreprocessor import BasePreprocessor


class SummarizePreprocessor(BasePreprocessor):
    def __init__(self, tokenizer: T5TokenizerFast, max_input_length=512, max_target_length=128):
        super().__init__(tokenizer)
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    def __call__(self, sample):
        inputs = sample["document"]
        labels = self.tokenizer(sample["summary"], max_length=self.max_target_length, truncation=True,
                                text_target=sample["summary"])

        model_inputs = self.tokenizer(inputs, max_length=self.max_input_length,
                                      truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
