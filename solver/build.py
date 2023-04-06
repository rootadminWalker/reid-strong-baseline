# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import torch

from .lr_scheduler import WarmupLR, DirectSetLR
from .fine_tune import FineTuning


def make_optimizer_with_center(cfg, model, center_criterion):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    if cfg.SOLVER.OPTIMIZER_NAME == 'SGD':
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params, momentum=cfg.SOLVER.MOMENTUM)
    else:
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params)
    optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=cfg.SOLVER.CENTER_LR)
    return optimizer, optimizer_center


def build_scheduler(optimizer, cfg):
    if cfg.SOLVER.LR_SCHEDULER_NAME == "cosine_annealing":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, cfg.SOLVER.MAX_EPOCHS, eta_min=cfg.SOLVER.MIN_LR
        )
    elif cfg.SOLVER.LR_SCHEDULER_NAME == "multistep_lr":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=cfg.SOLVER.STEPS,
            gamma=cfg.SOLVER.GAMMA, verbose=True
        )
    else:
        raise NotImplementedError(f"No such scheduler {cfg.SOLVER.LR_SCHEDULER_NAME}")

    return lr_scheduler


def build_warmup_lr(cfg):
    return WarmupLR(
        base_lr=cfg.SOLVER.BASE_LR,
        warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
        warmup_iters=cfg.SOLVER.WARMUP_ITERS,
        warmup_method=cfg.SOLVER.WARMUP_METHOD
    )


def build_direct_set_lr(cfg):
    return DirectSetLR(
        direct_steps=cfg.SOLVER.DIRECT_STEPS,
        direct_lrs=cfg.SOLVER.DIRECT_LRS
    )


def build_fine_tune(cfg):
    return FineTuning(unfreeze_at_epoch=cfg.SOLVER.UNFREEZE_AT_EPOCH)
