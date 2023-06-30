import numpy as np

from evaluate import load
from transformers import T5TokenizerFast

from NLP.src.utils.metrics.BaseMetrics import BaseMetrics


class TranslateMetrics(BaseMetrics):
    def __init__(self, tokenizer: T5TokenizerFast):
        super().__init__(tokenizer)

        self.__metric = load("sacrebleu")

    def __call__(self, eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds, decoded_labels = self.__postprocess_text(decoded_preds, decoded_labels)

        result = self.__metric.compute(predictions=decoded_preds, references=decoded_labels)
        result = {"bleu": result["score"]}

        prediction_lens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result

    @staticmethod
    def __postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]

        return preds, labels
