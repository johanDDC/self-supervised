import argparse
import os

from itertools import chain

import torch
import wandb

from transformers import T5TokenizerFast, AutoConfig, T5ForConditionalGeneration
from datasets import load_dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from datasets import Dataset

from src.models.T5DecoderWithHead import T5DecoderWithHead
from src.models.T5EncoderWithHead import T5EncoderWithHead
from src.utils.T5ForQA import T5ForQA


def define_config(task_name, ft=False, model_pretrain_path=None, model_warmup=0):
    if task_name not in ["qa", "summarization", "translation"]:
        raise NotImplementedError("This downstream task is not implemented")
    if not ft:
        if task_name == "qa":
            from configs.supervised.QAConfig import QAConfig
            return QAConfig()
        elif task_name == "summarization":
            from configs.supervised.SummarizationConfig import SummarizationConfig
            return SummarizationConfig()
        else:
            from configs.supervised.TranslationConfig import TranslationConfig
            return TranslationConfig()
    else:
        if task_name == "qa":
            from configs.fine_tune.TuneQA import TuneQA
            return TuneQA(pretrain_model_path=model_pretrain_path,
                          model_warmup=model_warmup)
        elif task_name == "summarization":
            from configs.fine_tune.TuneSummarization import TuneSummarization
            return TuneSummarization(pretrain_model_path=model_pretrain_path,
                                     model_warmup=model_warmup)
        else:
            from configs.fine_tune.TuneTranslation import TuneTranslation
            return TuneTranslation(pretrain_model_path=model_pretrain_path,
                                   model_warmup=model_warmup)


def prepare_fine_tune(target_model, freeze=False):
    state_dict = torch.load(config.pretrain_model_path)
    if pretext_model == "gpt":
        del state_dict["lm_head.bias"]
    target_model.load_state_dict(state_dict)
    freezed_params = None
    if freeze:
        if downstream_task == "qa":
            for p in target_model.t5.parameters():
                p.requires_grad_(False)
            freezed_params = target_model.t5.parameters()
        else:
            for p in chain(target_model.decoder.parameters(), target_model.lm_head.parameters()):
                p.requires_grad_(False)
            freezed_params = chain(target_model.decoder.parameters(), target_model.lm_head.parameters())
    return target_model, freezed_params


def define_translation(config):
    from src.utils.collators.BaseCollator import BaseCollator, BaseTestCollator
    from src.utils.metrics.TranslateMetrics import TranslateMetrics
    from src.utils.preprocessors.TranslatePreprocessor import TranslatePreprocessor
    from src.utils.wrappers.TranslateWrapper import TranslateWrapper

    max_input_length = config.max_input_length
    max_target_length = config.max_target_length
    source_lang = config.source_lang
    target_lang = config.target_lang

    preprocessor = TranslatePreprocessor(tokenizer, source_lang, target_lang,
                                         max_input_length, max_target_length)
    collator = BaseCollator(tokenizer)
    test_collator = BaseTestCollator(tokenizer)

    dataset = load_dataset(config.dataset_name, config.dataset_info)
    dataset["train"] = dataset["train"].select(
        range(int(len(dataset['train']) * config.dataset_split / 100))
    )

    model = T5ForConditionalGeneration(model_cfg)
    freezed_params = None
    if ft:
        model, freezed_params = prepare_fine_tune(model, model_warmup > 0)
    model = TranslateWrapper(model, device, log_steps)

    metric_dict = {
        "bleu": TranslateMetrics(tokenizer)
    }
    return preprocessor, collator, test_collator, dataset, model, freezed_params, metric_dict


def define_summarization(config):
    from src.utils.collators.BaseCollator import BaseCollator, BaseTestCollator
    from src.utils.metrics.SummarizeMetrics import SummarizeMetrics
    from src.utils.preprocessors.SummarizePreprocessor import SummarizePreprocessor
    from src.utils.wrappers.SummarizeWrapper import SummarizeWrapper

    max_input_length = config.max_input_length
    max_target_length = config.max_target_length
    preprocessor = SummarizePreprocessor(tokenizer, max_input_length, max_target_length)
    collator = BaseCollator(tokenizer)
    test_collator = BaseTestCollator(tokenizer)

    train_dataset = load_dataset(config.dataset_name, split=f"train[:{config.dataset_split}%]")
    val_dataset = load_dataset(config.dataset_name, split="validation")
    dataset = {"train": train_dataset, "validation": val_dataset}

    model = T5ForConditionalGeneration(model_cfg)
    freezed_params = None
    if ft:
        model, freezed_params = prepare_fine_tune(model, model_warmup > 0)
    model = SummarizeWrapper(model, device, log_steps)

    metric_dict = {
        "rouge1": SummarizeMetrics(tokenizer)
    }
    return preprocessor, collator, test_collator, dataset, model, freezed_params, metric_dict


def define_qa(config):
    from src.utils.collators.QACollator import QACollator
    from src.utils.metrics.QAMetrics import QAMetrics
    from src.utils.preprocessors.QAPreprocessor import QAPreprocessor
    from src.utils.wrappers.QAWrapper import QAWrapper

    max_length = config.max_length
    doc_stride = config.doc_stride

    preprocessor = QAPreprocessor(tokenizer, max_length, doc_stride)
    collator = QACollator(tokenizer)

    dataset = load_dataset("squad")

    model = T5ForQA(model_cfg)
    freezed_params = None
    if ft:
        model, freezed_params = prepare_fine_tune(model, model_warmup > 0)
    model = QAWrapper(model, device, log_steps)

    metric_dict = {
        "accuracy": QAMetrics(tokenizer, "accuracy"),
        "f1": QAMetrics(tokenizer, "f1"),
    }
    return preprocessor, collator, collator, dataset, model, freezed_params, metric_dict


def define_downstream_task(task_name, config):
    task_name = task_name.lower()
    if task_name == "qa":
        return define_qa(config)
    elif task_name == "summarization":
        return define_summarization(config)
    elif task_name == "translation":
        return define_translation(config)


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


def get_dataset_map_params(task_name):
    if task_name == "qa":
        return {"remove_columns": dataset["train"].column_names}
    return {}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--downstream_task', type=str, help="Downstream task", required=True)
    parser.add_argument('--pretext_model', type=str, help="Type of pretrain task", default="")
    parser.add_argument('--fine_tune', help="Fine tuning", action="store_true")
    parser.add_argument('--model_pretrain_path', type=str, help="Path to pretrained model", default="")
    parser.add_argument('--model_warmup', type=float, help="ratio of steps before model fully unfreezed", default=0.)
    parser.add_argument('--num_epochs', type=int, help="Num epochs", default=1)
    parser.add_argument('--accum_steps', type=int, help="Accumulation Steps", default=1)
    parser.add_argument('--batch_size', type=int, help="Batch Size", default=16)
    parser.add_argument('--log_steps', type=int, help="Log steps", default=500)
    args = dict(vars(parser.parse_args()))

    downstream_task = args["downstream_task"]
    pretext_model = args["pretext_model"]
    num_epochs = args["num_epochs"]
    batch_size = args["batch_size"]
    accum_steps = args['accum_steps']
    log_steps = args['log_steps']
    ft = args["fine_tune"]
    model_pretrain_path = args["model_pretrain_path"]
    model_warmup = args["model_warmup"]

    config = define_config(downstream_task, ft, model_pretrain_path, model_warmup)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = T5TokenizerFast.from_pretrained(config.model_name)
    model_cfg = AutoConfig.from_pretrained(config.model_name)
    model_cfg.feed_forward_proj = "gated_gelu"

    preprocessor, collator, test_collator, dataset, model, freezed_params, metric_dict = \
        define_downstream_task(downstream_task, config)

    if type(dataset) is dict:
        train_dataset = dataset["train"].map(preprocessor, batched=True, num_proc=4, batch_size=5000,
                                             **get_dataset_map_params(downstream_task))
        val_dataset = dataset["validation"].map(preprocessor, batched=True, num_proc=4, batch_size=5000,
                                                **get_dataset_map_params(downstream_task))
        dataset = {"train": train_dataset, "validation": val_dataset}
    else:
        dataset = dataset.map(preprocessor, batched=True, num_proc=4, batch_size=5000,
                              **get_dataset_map_params(downstream_task))
    train_loader = DataLoader(
        dataset["train"],
        collate_fn=collator,
        batch_size=batch_size,
        pin_memory=True, shuffle=True,
        num_workers=6
    )
    val_loader = DataLoader(
        dataset["validation"],
        collate_fn=test_collator,
        batch_size=batch_size,
        pin_memory=True, shuffle=False,
        num_workers=6
    )

    num_train_steps = len(train_loader) * num_epochs // accum_steps
    model.setup_fine_tune(int(model_warmup * num_train_steps), freezed_params)

    optimizer = AdamW(model.parameters(), **get_optimizer_params(config))
    scheduler = OneCycleLR(optimizer, **get_scheduler_params(config),
                           total_steps=num_train_steps + 5)

    os.makedirs(f"checkpoints/{downstream_task}/train_checkpoints/", exist_ok=True)
    os.makedirs(f"checkpoints/{downstream_task}/final_model/", exist_ok=True)
    with wandb.init(project=config.project_name, entity=config.entity_name,
                    name=config.run_name):
        best_metric = 0
        best_model = None
        for epoch in range(1, num_epochs + 1):
            epoch_loss = model.train_epoch(train_loader, optimizer, scheduler, accum_steps)
            metrics = model.evaluate(val_loader, metric_dict)

            log_metrics = {k: metrics[v] for k, v in
                           zip(config.log_keys, config.log_ids)}

            wandb.log({
                "epoch_loss": epoch_loss.item(),
                **log_metrics
            })

            torch.save({
                "model": model.model.state_dict(),
                "opt": optimizer.state_dict(),
                "scheduler": scheduler.state_dict()
            }, f"checkpoints/{downstream_task}/train_checkpoints/{epoch}.pth")

            if best_model is None or metrics[config.key_metric] > best_metric:
                best_metric = metrics[config.key_metric]
                best_model = model.model.state_dict()

        torch.save(best_model, f"checkpoints/{downstream_task}/final_model/final_model.pth")
