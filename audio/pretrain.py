import os
import argparse

import wandb
import torch
import torch.nn as nn
import numpy as np
import torchaudio.transforms as T

from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.collate import crop_collator
from src.data.dataset import AudioDataset
from src.model.MFCL.MFCL import MFCL
from src.model.MFCL.loss import ContrastiveLoss
from src.utils.preprocessing import form_df
from src.utils.utils import set_random_seed

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
            logits = model(batch["wav_crops"], batch["mel_crops"])
            loss = criterion(torch.vstack(logits))
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

        losses += loss.detach() * batch["speaker"].shape[0]

    return losses / len(dataloader.dataset)


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    losses = torch.zeros((1,), device=device)

    model.eval()
    for batch in tqdm(dataloader, total=len(dataloader)):
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

        logits = model(batch["wav_crops"], batch["mel_crops"])
        loss = criterion(torch.vstack(logits))
        losses += loss.detach() * batch["speaker"].shape[0]

    return losses / len(dataloader.dataset)


def get_dataloaders(data_dir, batch_size=32, num_workers=6):
    speaker_ids = np.array(os.listdir(data_dir))

    test_speakers = np.random.choice(speaker_ids, int(len(speaker_ids) // 3), False)
    train_val_speakers = np.array([idx for idx in speaker_ids if idx not in test_speakers])
    train_val_df = form_df(train_val_speakers, data_dir)
    train_val_df = train_val_df.sample(frac=1, random_state=SEED)
    test_df = form_df(test_speakers, data_dir)

    train_dataset = AudioDataset(train_val_df)
    test_dataset = AudioDataset(test_df)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                  collate_fn=lambda batch: crop_collator(batch, ratio=0.6, overlap=True), pin_memory=True, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                 collate_fn=lambda batch: crop_collator(batch, ratio=0.6, overlap=True), pin_memory=True, drop_last=True)

    return train_dataloader, test_dataloader


def get_augmentations(param):
    if param == "no":
        return (nn.Identity(), nn.Identity())
    elif param == "wav":
        return (T.TimeMasking(time_mask_param=15), nn.Identity())
    elif param == "mel":
        return (nn.Identity(), T.FrequencyMasking(freq_mask_param=15))
    else:
        return (T.TimeMasking(time_mask_param=15),
                T.FrequencyMasking(freq_mask_param=15))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--augmentations", type=str, default="no")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--log_steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=322)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    augmentations = args.augmentations
    batch_size = args.batch_size
    num_workers = args.num_workers
    num_epochs = args.num_epochs
    max_log_steps = args.log_steps
    seed = args.seed
    device = args.device

    set_random_seed(seed)

    data_dir = "data/wavs"
    train_dataloader, test_dataloader = get_dataloaders(data_dir)
    transforms = get_augmentations(augmentations)
    model = MFCL(transforms, device, min_eps=1e-6, max_eps=1e4, sample_rate=16_000, n_fft=400, win_length=400,
                 hop_length=160, n_mels=80)
    criterion = ContrastiveLoss(temperature=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, num_epochs * len(train_dataloader) + 2, 1e-6)

    model.to(device)
    os.makedirs(f"checkpoints/mfcl/", exist_ok=True)
    with wandb.init(job_type="supervised", dir="wandb_logs", project="ssl_hw4", entity="johan_ddc_team",
                    name=f"MFCL_{augmentations}_augs"):
        for i in range(1, num_epochs + 1):
            epoch_loss = train_one_epoch(model, train_dataloader, criterion, optimizer, device, scheduler=scheduler)
            val_loss = evaluate(model, test_dataloader, criterion, device)

            wandb.log({
                "epoch_loss": epoch_loss.item(),
                "test_loss": val_loss.item(),
            })
        torch.save(model.state_dict(), f"checkpoints/mfcl/{augmentations}.pth")
