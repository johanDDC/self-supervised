from evaluate import load
from transformers import T5TokenizerFast

from NLP.src.utils.metrics.BaseMetrics import BaseMetrics


class QAMetrics(BaseMetrics):
    def __init__(self, tokenizer: T5TokenizerFast, metric_name):
        super().__init__(tokenizer)

        self.metric_name = metric_name
        self.__metric = load(self.metric_name)

    def __call__(self, eval_preds):
        start_logits, end_logits, start_positions, end_positions = eval_preds
        kwargs_start = {"predictions": start_logits.argmax(axis=-1), "references": start_positions}
        kwargs_end = {"predictions": end_logits.argmax(axis=-1), "references": end_positions}
        if self.metric_name == "f1":
            kwargs_start["average"] = kwargs_end["average"] = "weighted"

        res = {
            self.metric_name: (
                self.__metric.compute(**kwargs_start)[self.metric_name] +
                self.__metric.compute(**kwargs_end)[self.metric_name]
            ) / 2
        }
        return res
