from transformers import T5TokenizerFast

from NLP.src.utils.preprocessors.BasePreprocessor import BasePreprocessor


class TranslatePreprocessor(BasePreprocessor):
    def __init__(self, tokenizer: T5TokenizerFast, source_lang, target_lang,
                 max_input_length=512, max_target_length=128):
        super().__init__(tokenizer)
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.source_lang = source_lang
        self.target_lang = target_lang

    def __call__(self, sample):
        inputs = [ex[self.source_lang] for ex in sample["translation"]]
        targets = [ex[self.target_lang] for ex in sample["translation"]]
        labels = self.tokenizer(targets, max_length=self.max_target_length,
                                truncation=True, text_target=targets)

        model_inputs = self.tokenizer(inputs, max_length=self.max_input_length,
                                      truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
