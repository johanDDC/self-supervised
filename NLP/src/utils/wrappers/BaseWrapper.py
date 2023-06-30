import wandb
import torch
import torch.nn as nn
import numpy as np

from typing import Dict, Union
from collections import defaultdict
from abc import ABC, abstractmethod

from tqdm import tqdm

from NLP.src.utils.metrics.BaseMetrics import BaseMetrics


class BaseWrapper(ABC):
    def __init__(self, model: nn.Module, device, log_steps=100):
        self.model = model
        self.device = device
        self.log_steps = log_steps

        self.__steps = 0
        self.__scaler = torch.cuda.amp.GradScaler()
        self.__ft = False

        self.model.to(device)

    @property
    def parameters(self):
        return self.model.parameters

    @abstractmethod
    def loss(self, batch):
        pass

    def setup_fine_tune(self, warmup_steps, freezed_params):
        self.__warmup_steps = 0
        self.__max_warmup_steps = warmup_steps
        self.__freezed_params = freezed_params
        self.__ft = True

    def train_epoch(self, dataloader, optimizer, scheduler=None,
                    accum_steps=1, max_grad_norm=1):
        epoch_loss = torch.zeros((1,), device=self.device)

        self.model.train()
        pbar = tqdm(enumerate(dataloader), total=len(dataloader))
        for idx, batch in pbar:
            if idx >= len(dataloader):
                break
            batch = {k: v.to(self.device, non_blocking=True)
                     for k, v in batch.items()}

            with torch.autocast(device_type=self.device, dtype=torch.float16):
                loss = self.loss(batch)
            self.__scaler.scale(loss / accum_steps).backward()

            if idx % accum_steps == accum_steps - 1:
                self.__scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.parameters(),
                                               max_norm=max_grad_norm)
                self.__scaler.step(optimizer)
                self.__scaler.update()
                optimizer.zero_grad(set_to_none=True)

                if scheduler is not None:
                    scheduler.step()

            epoch_loss += loss.detach()
            self.__steps += 1
            if self.log_steps > 0 and self.__steps % self.log_steps == 0:
                wandb.log({
                    "loss": loss.item(),
                    "lr": optimizer.param_groups[0]["lr"]
                })
                self.__steps = 1
            if self.__ft:
                self.__warmup_steps += 1
                if self.__warmup_steps >= self.__max_warmup_steps:
                    for p in self.__freezed_params:
                        p.requires_grad_(True)

        return epoch_loss / len(dataloader)

    @torch.no_grad()
    def evaluate(self, dataloader, metric_dict=Union[None, Dict[str, BaseMetrics]]):
        metrics = defaultdict(list)
        self.model.eval()

        pbar = tqdm(dataloader, total=len(dataloader))
        for batch, target in pbar:
            batch = {k: v.to(self.device, non_blocking=True)
                     for k, v in batch.items()}

            with torch.autocast(device_type=self.device, dtype=torch.float16):
                output_sequences = self.model.generate(
                    **batch, do_sample=False
                )

            if metric_dict is not None:
                for k, v in metric_dict.items():
                    metric = v([output_sequences.cpu(), target])[k]
                    metrics[k].append(metric)

        for k, v in metrics.items():
            metrics[k] = np.mean(v)
        return metrics
