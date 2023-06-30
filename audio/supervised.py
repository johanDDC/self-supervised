import os
import argparse

import wandb
import torch
import torch.nn as nn
import numpy as np

from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from pretrain import get_augmentations
from src.data.dataset import AudioDataset
from src.model.Encoder1D import Encoder1D
from src.model.Encoder2D import Encoder2D
from src.model.MFCL.MFCL import MFCL
from src.utils.ConcatOutput import ConcatOutput
from src.utils.LogMelSpec import LogMelSpec
from src.utils.preprocessing import form_df
from src.utils.utils import set_random_seed

SEED = 322
log_steps = 0


def train_one_epoch(model, dataloader, criterion, optimizer, device, scheduler=None, max_log_steps=50):
    global log_steps
    losses = torch.zeros((1,), device=device)
    scaler = torch.cuda.amp.GradScaler()

    model.train()
    pbar = tqdm(dataloader, total=len(dataloader))
    for idx, batch in enumerate(pbar):
        if idx >= len(pbar):
            break
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

        with torch.autocast(device_type=device, dtype=torch.float16):
            logits = model(batch["wav"])
            loss = criterion(logits, batch["target"])
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                       max_norm=1.)
        scaler.step(optimizer)
        scaler.update()

        optimizer.zero_grad(set_to_none=True)

        if scheduler is not None:
            scheduler.step()

        log_steps += 1
        if log_steps >= max_log_steps:
            wandb.log({"lr": optimizer.param_groups[0]["lr"]})
            log_steps = 0

        losses += loss.detach() * batch["wav"].shape[0]

    return losses / len(dataloader.dataset)


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    losses = torch.zeros((1,), device=device)
    accuracy = torch.zeros((1,), device=device)

    model.eval()
    for batch in tqdm(dataloader, total=len(dataloader)):
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

        logits = model(batch["wav"])
        loss = criterion(logits, batch["target"])

        losses += loss.detach() * batch["wav"].shape[0]
        accuracy += (logits.argmax(dim=1) == batch["target"]).sum()

    return losses / len(dataloader.dataset), accuracy / len(dataloader.dataset)


def load_model(augs_type, device="cuda"):
    if len(augs_type) == 0:
        if model_type == "1d":
            return nn.Sequential(
                Encoder1D(),
                nn.Linear(512, 10)
            )
        elif model_type == "2d":
            return nn.Sequential(
                LogMelSpec(min_eps=1e-6, max_eps=1e4, sample_rate=16_000, n_fft=400, win_length=400,
                           hop_length=160, n_mels=80),
                Encoder2D(),
                nn.Linear(512, 10)
            )
        else:
            raise NotImplementedError()
    transforms = get_augmentations(augs_type)
    mdl = MFCL(transforms, device, min_eps=1e-6, max_eps=1e4, sample_rate=16_000, n_fft=400, win_length=400,
               hop_length=160, n_mels=80)
    mdl.load_state_dict(torch.load(f"checkpoints/mfcl/{augs_type}.pth"))
    for p in mdl.parameters():
        p.requires_grad_(False)
    if augs_type == "wav":
        return nn.Sequential(
            mdl.wav_encoder,
            nn.Linear(512, 10)
        )
    elif augs_type == "mel":
        return nn.Sequential(
            LogMelSpec(min_eps=1e-6, max_eps=1e4, sample_rate=16_000, n_fft=400, win_length=400,
                       hop_length=160, n_mels=80),
            mdl.spec_encoder,
            nn.Linear(512, 10)
        )
    else:
        return nn.Sequential(
            mdl,
            ConcatOutput(),
            nn.Linear(1024, 10)
        )


def get_dataloaders(data_dir, batch_size=32, num_workers=6):
    np.random.seed(SEED)
    speaker_ids = np.array(os.listdir(data_dir))

    test_speakers = np.random.choice(speaker_ids, int(len(speaker_ids) // 3), False)
    train_val_speakers = np.array([idx for idx in speaker_ids if idx not in test_speakers])
    train_val_df = form_df(train_val_speakers, data_dir)
    train_val_df = train_val_df.sample(frac=1, random_state=SEED)

    groups = train_val_df.groupby(by=["speaker", "y"])
    train_split = 0.8

    train_ids = np.array([])
    val_ids = np.array([])
    for i, group in groups:
        group_train_ids, group_val_ids = np.split(group.index, [int(len(group) * train_split)])
        train_ids = np.append(train_ids, group_train_ids)
        val_ids = np.append(val_ids, group_val_ids)

    train_df = train_val_df.iloc[train_ids]
    val_df = train_val_df.iloc[val_ids]
    test_df = form_df(test_speakers, data_dir)

    train_dataset = AudioDataset(train_df)
    val_dataset = AudioDataset(val_df)
    test_dataset = AudioDataset(test_df)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                  collate_fn=train_dataset.collator, pin_memory=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                collate_fn=val_dataset.collator, pin_memory=True, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                 collate_fn=test_dataset.collator, pin_memory=True, drop_last=True)

    return train_dataloader, val_dataloader, test_dataloader


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, default="1d")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--log_steps", type=int, default=25)
    parser.add_argument("--augs_type", type=str, default="")
    args = parser.parse_args()

    model_type = args.type
    batch_size = args.batch_size
    num_workers = args.num_workers
    num_epochs = args.num_epochs
    max_log_steps = args.log_steps
    augs_type = args.augs_type

    set_random_seed(SEED)
    device = "cuda"

    data_dir = "data/wavs"
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(data_dir)
    criterion = nn.CrossEntropyLoss()
    model = load_model(augs_type)

    if len(augs_type) == 0:
        optimizer = torch.optim.AdamW(model.parameters(), lr=4e-3, betas=(0.9, 0.999), weight_decay=0.01)
        scheduler = CosineAnnealingLR(optimizer, num_epochs * len(train_dataloader) + 2, 1e-6)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        scheduler = CosineAnnealingLR(optimizer, num_epochs * len(train_dataloader) + 2, 1e-6)

    model.to(device)
    os.makedirs(f"checkpoints/{model_type}{'_' + augs_type if len(augs_type) > 0 else ''}/", exist_ok=True)
    best = [None, 0]
    with wandb.init(job_type="supervised", dir="wandb_logs", project="ssl_hw4", entity="johan_ddc_team",
                    name=f"encoder_{model_type}_supervised"):
        for i in range(1, num_epochs + 1):
            epoch_loss = train_one_epoch(model, train_dataloader, criterion, optimizer, device, scheduler=scheduler)
            val_loss, val_acc = evaluate(model, test_dataloader, criterion, device)

            wandb.log({
                "epoch_loss": epoch_loss.item(),
                "test_loss": val_loss.item(),
                "accuracy": val_acc.item()
            })

            if best[0] is None or val_acc > best[1]:
                best = [model.state_dict(), val_acc]
        torch.save(best[0], f"checkpoints/{model_type}{'_' + augs_type if len(augs_type) > 0 else ''}/final_model.pth")
