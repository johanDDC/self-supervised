import os
import argparse

import torch
import wandb

from datasets import load_dataset, load_from_disk
from transformers import T5TokenizerFast
from torch.utils.data import DataLoader

from src.models.GPT import GPT
from src.models.MASS import MASS
from src.models.MLM import MLM
from src.utils.FixedTokensSampler import FixedTokensSampler, collate_batch
from src.utils.collators.MASSCollator import MASSCollator
from src.utils.collators.MLMCollator import MLMCollator


def define_config(task_name):
    if task_name == "mlm":
        from configs.pretext.MLMConfig import MLMConfig
        return MLMConfig()
    elif task_name == "gpt":
        from configs.pretext.GPTConfig import GPTConfig
        return GPTConfig()
    elif task_name == "mass":
        from configs.pretext.MASSConfig import MASSConfig
        return MASSConfig()
    else:
        raise NotImplementedError("This pretext task is not implemented")


def define_model(task_name, config):
    if task_name == "mlm":
        model = MLM(model_name, device,
                    tokenizer.pad_token_id, log_steps)
        collator = MLMCollator(tokenizer)
        return model, collator
    elif task_name == "gpt":
        return GPT(model_name, device, log_steps), \
            collate_batch
    else:
        return MASS(model_name, device, tokenizer.pad_token_id,
                    log_steps), MASSCollator(tokenizer, config.one_seq)


def get_optimizer_params(config):
    return {
        "lr": config.lr,
        "weight_decay": config.weight_decay,
        "betas": config.betas
    }


def get_scheduler_params(config):
    return {
        "max_lr": config.lr,
        "pct_start": config.pct_start,
        "anneal_strategy": config.anneal_strategy,
        "div_factor": config.div_factor
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretext_task', type=str, help="Pretext task", required=True)
    parser.add_argument('--num_epochs', type=int, help="Num epochs", default=1)
    parser.add_argument('--accum_steps', type=int, help="Accumulation Steps", default=1)
    parser.add_argument('--batch_size', type=int, help="Batch Size", default=16)
    parser.add_argument('--log_steps', type=int, help="Log steps", default=500)
    args = dict(vars(parser.parse_args()))

    pretext_tast = args["pretext_task"]
    num_epochs = args["num_epochs"]
    batch_size = args["batch_size"]
    accum_steps = args['accum_steps']
    log_steps = args['log_steps']

    config = define_config(pretext_tast)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = config.model_name
    tokenizer = T5TokenizerFast.from_pretrained(model_name)
    if not os.path.exists("data/encoded_bookcorpus"):
        dataset = load_dataset("bookcorpus", split="train", num_proc=8)
        dataset = dataset.select(range(int(len(dataset) * 0.1)))
        encoded_dataset = dataset.map(lambda examples:
                                      tokenizer(examples['text'],
                                                truncation=True,
                                                max_length=config.max_length),
                                      batched=True, num_proc=4, batch_size=5000)
        encoded_dataset.save_to_disk("data/encoded_bookcorpus")
    else:
        encoded_dataset = load_from_disk("data/encoded_bookcorpus")

    sampler = FixedTokensSampler(encoded_dataset, n_tokens=int(2 ** 13), shuffle=True)
    model, collator = define_model(pretext_tast, config)
    dataloader = DataLoader(
        encoded_dataset,
        batch_sampler=sampler,
        collate_fn=collator,
        num_workers=6,
        pin_memory=True
    )

    num_train_steps = len(dataloader) * num_epochs // accum_steps
    optim = torch.optim.AdamW(model.parameters(),
                              **get_optimizer_params(config))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optim, **get_scheduler_params(config),
                                                    total_steps=num_train_steps + 5)

    os.makedirs(f"checkpoints/{pretext_tast}/train_checkpoints/", exist_ok=True)
    os.makedirs(f"checkpoints/{pretext_tast}/final_model/", exist_ok=True)
    with wandb.init(project=config.project_name, entity=config.entity_name,
                    name=config.run_name):
        for epoch in range(1, num_epochs + 1):
            epoch_loss = model.train_epoch(dataloader,
                                           optim, scheduler,
                                           accum_steps)

            wandb.log({
                "epoch_loss": epoch_loss.item()
            })

            torch.save({
                "model": model.model.state_dict(),
                "opt": optim.state_dict(),
                "scheduler": scheduler.state_dict()
            }, f"checkpoints/{pretext_tast}/train_checkpoints/{epoch}.pth")

        torch.save(model.model.state_dict(),
                   f"checkpoints/{pretext_tast}/final_model/mlm_wrapped.pth")
        torch.save(model.to_downstream_task("qa").state_dict(),
                   f"checkpoints/{pretext_tast}/final_model/downstream_qa.pth")
        torch.save(model.to_downstream_task("translate").state_dict(),
                   f"checkpoints/{pretext_tast}/final_model/downstream_seq2seq.pth")
