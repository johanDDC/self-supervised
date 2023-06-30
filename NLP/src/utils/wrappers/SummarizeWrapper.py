from NLP.src.utils.wrappers.BaseWrapper import BaseWrapper


class SummarizeWrapper(BaseWrapper):
    def loss(self, batch):
        outputs = self.model(
            input_ids=batch["input_ids"],
            labels=batch["labels"],
            attention_mask=batch["attention_mask"]
        )
        return outputs.loss
