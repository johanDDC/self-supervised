import argparse
import os

import torch
import warnings
import torch.nn as nn
import wandb

from torch.utils.data import DataLoader
import torchvision.models as mdls
from tqdm import tqdm

import CV.models.SimCLR
import utils
import configs
import models
from CV.data.data import get_train_data, get_pretrain_data

warnings.filterwarnings("ignore")


def train_epoch(model, optimizer, criterion, loader, mixup_alpha):
    model.train()
    train_loss = torch.zeros(1, device=device, dtype=torch.float32)
    pbar = tqdm(loader, total=len(loader))
    for batch_id, (batch, target) in enumerate(pbar):
        batch = batch.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        batch, target_a, target_b, lmbd = utils.mixup_data(batch, target, mixup_alpha, device)

        preds = model(batch)
        loss = utils.mixup_criterion(criterion, preds, target_a, target_b, lmbd)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        train_loss += loss.detach()
        pbar.update(1)
    return train_loss / len(loader)


@torch.no_grad()
def evaluate(model, criterion, loader):
    model.eval()
    accuracy = 0
    val_loss = torch.zeros(1, device=device, dtype=torch.float32)
    pbar = tqdm(loader, total=len(loader))
    for batch_id, (batch, target) in enumerate(pbar):
        batch = batch.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        preds = model(batch)
        loss = criterion(preds, target)

        val_loss += loss.detach()
        accuracy += (preds.argmax(dim=1) == target).float().mean()
        pbar.set_description(f"accuracy: {accuracy / ((batch_id + 1))}")
        pbar.update(1)
    return val_loss / len(loader), accuracy / len(loader)


def train(cfg, model, optimizer, criterion, train_loader, test_loader, scheduler=None):
    model.to(device)
    best_accuracy = 0
    for epoch in range(1, cfg.num_epochs + 1):
        epoch_loss = train_epoch(model, optimizer, criterion, train_loader, mixup_alpha=cfg.mixup_alpha)
        val_loss, val_accuracy = evaluate(model, criterion, test_loader)

        wandb.log({
            "train_loss": epoch_loss.cpu(),
            "val_loss": val_loss.cpu(),
            "lr": optimizer.param_groups[0]["lr"],
            "accuracy": val_accuracy.cpu()
        })

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }, os.path.join("checkpoints", f"{mode}.pth"))

        if scheduler is not None:
            scheduler.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, help="Train mode", required=True)
    parser.add_argument('--seed', type=int, help="Random seed", default=322)
    parser.add_argument('--nw', type=int, help="Num workers", default=6)
    ft_parser = parser.add_mutually_exclusive_group(required=False)
    ft_parser.add_argument('--ft', dest="ft", action="store_true", help="Fine tune pretrained model")
    ft_parser.add_argument('--pb', dest="ft", action='store_false', help="Use linear probing for pretrained model")
    parser.set_defaults(ft=False)
    parser.add_argument('--device', type=str, help="Device", default="cuda")
    args = dict(vars(parser.parse_args()))
    mode, seed, num_workers, ft, device = args["mode"], args["seed"], args["nw"], args["ft"], args["device"]

    utils.set_random_seed(seed)

    if mode == "supervised":
        cfg = configs.supervised.SupervisedConfig()
        model = mdls.resnet18(num_classes=10)
    elif mode == "SimCLR":
        cfg = configs.SimCLR.SimCLRConfig()
        model_manager = CV.models.SimCLR
    elif mode == "BYOL":
        cfg = configs.BYOL.BYOLConfig()
        model_manager = models.BYOL
    else:
        raise NotImplementedError("Such train mode is not implemented")

    if mode != "supervised":
        pretrain_data = get_pretrain_data(root_dir="data")
        model = model_manager.pretrain(cfg, pretrain_data, num_workers=num_workers, device=device)
        cfg = configs.supervised.SupervisedConfig()
        if not ft:
            for p in model.parameters():
                p.requires_grad_(False)
        model = utils.Probing(model)

    stl10_train, stl10_test = get_train_data(root_dir="data")
    train_loader = DataLoader(stl10_train, batch_size=cfg.batch_size, shuffle=True,
                              pin_memory=True, num_workers=num_workers, drop_last=True)
    test_loader = DataLoader(stl10_test, batch_size=cfg.batch_size, shuffle=False,
                             pin_memory=True, num_workers=num_workers)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, cfg.scheduler_milestones)
    criterion = nn.CrossEntropyLoss()

    os.makedirs("checkpoints", exist_ok=True)
    with wandb.init(project=cfg.project_name, entity=cfg.entity_name, name=cfg.model_name) as run:
        wandb.watch(model, optimizer, log="all", log_freq=10)
        train(cfg, model, optimizer, criterion, train_loader, test_loader, scheduler)
        torch.save(model.state_dict(), os.path.join("checkpoints", f"{mode}_final.pth"))
