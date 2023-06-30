import torch
import numpy as np

from typing import Union, Dict
from collections import defaultdict

from tqdm import tqdm

from NLP.src.utils.metrics.BaseMetrics import BaseMetrics
from NLP.src.utils.wrappers.BaseWrapper import BaseWrapper


class QAWrapper(BaseWrapper):
    def loss(self, batch):
        outputs = self.model(
            input_ids=batch["input_ids"],
            start_positions=batch["start_positions"],
            end_positions=batch["end_positions"],
            attention_mask=batch["attention_mask"]
        )
        return outputs.loss

    @torch.no_grad()
    def evaluate(self, dataloader, metric_dict=Union[None, Dict[str, BaseMetrics]]):
        metrics = defaultdict(list)
        self.model.eval()

        pbar = tqdm(dataloader, total=len(dataloader))
        for batch in pbar:
            batch = {k: v.to(self.device, non_blocking=True)
                     for k, v in batch.items()}

            with torch.autocast(device_type=self.device, dtype=torch.float16):
                output = self.model(
                    **batch
                )

            if metric_dict is not None:
                for k, v in metric_dict.items():
                    metric = v([output["start_logits"], output["end_logits"],
                                batch["start_positions"], batch["end_positions"]])[k]
                    metrics[k].append(metric)

        for k, v in metrics.items():
            metrics[k] = np.mean(v)
        return metrics
