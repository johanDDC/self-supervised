import nltk
import numpy as np

from evaluate import load
from transformers import T5TokenizerFast

from NLP.src.utils.metrics.BaseMetrics import BaseMetrics


class SummarizeMetrics(BaseMetrics):
    def __init__(self, tokenizer: T5TokenizerFast):
        super().__init__(tokenizer)

        self.__metric = load("rouge")

    def __call__(self, eval_preds):
        predictions, labels = eval_preds
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
        decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

        result = self.__metric.compute(predictions=decoded_preds,
                                       references=decoded_labels,
                                       use_stemmer=True)
        result = {key: value * 100 for key, value in result.items()}

        prediction_lens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)

        return {k: round(v, 4) for k, v in result.items()}
