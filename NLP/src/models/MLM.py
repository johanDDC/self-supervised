import torch
import torch.nn as nn
import wandb

from transformers import AutoConfig, T5ForConditionalGeneration
from tqdm import tqdm

from NLP.src.models.T5EncoderWithHead import T5EncoderWithHead
from NLP.src.utils.T5ForQA import T5ForQA


class MLM:
    def __init__(self, model_name, device, pad_token_id=0, log_steps=100):
        self.device = device
        self.model: T5EncoderWithHead = self.__init_model(model_name)
        self.log_steps = log_steps
        self.model_name = model_name

        self.__criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)
        self.__scaler = torch.cuda.amp.GradScaler()
        self.__steps = 0

        self.model.to(device)

    @staticmethod
    def __init_model(model_name):
        config = AutoConfig.from_pretrained(model_name)
        config.feed_forward_proj = "gated_gelu"
        model = T5EncoderWithHead(config, config.vocab_size)
        return model

    @property
    def parameters(self):
        return self.model.parameters

    def train_epoch(self, dataloader, optimizer, scheduler=None,
                    accum_steps=1, max_grad_norm=1):
        losses = torch.zeros((1,), device=self.device)

        self.model.train()
        pbar = tqdm(enumerate(dataloader), total=len(dataloader))
        for idx, (inpt, target) in pbar:
            if idx >= len(dataloader):
                break
            inpt, target = inpt.to(self.device, non_blocking=True), \
                target.to(self.device, non_blocking=True)

            with torch.autocast(device_type=self.device, dtype=torch.float16):
                preds = self.model(inpt)
                loss = self.__criterion(preds.transpose(1, 2), target)
            self.__scaler.scale(loss).backward()

            if idx % accum_steps == accum_steps - 1:
                self.__scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               max_norm=max_grad_norm)
                self.__scaler.step(optimizer)
                self.__scaler.update()
                optimizer.zero_grad(set_to_none=True)

                if scheduler is not None:
                    scheduler.step()

            losses += loss.detach() * inpt.shape[0]
            self.__steps += 1
            if self.__steps % self.log_steps == 0:
                wandb.log({
                    "loss": loss.item(),
                    "lr": optimizer.param_groups[0]["lr"]
                })
                self.__steps = 1

        return losses / len(dataloader.dataset)

    def to_downstream_task(self, task_name):
        config = AutoConfig.from_pretrained(self.model_name)
        if task_name == "qa":
            new_model = T5ForQA(config)
            new_model.t5 = self.model.encoder
        else:
            new_model = T5ForConditionalGeneration(config)
            new_model.encoder = self.model.encoder
            new_model.lm_head = self.model.head
        return new_model
