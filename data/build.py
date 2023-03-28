# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .collate_batch import train_collate_fn, test_collate_fn
from .datasets import init_dataset, ImageDataset
from .samplers import RandomIdentitySampler  # New add by gu
from .transforms import build_transforms_stage


class REIDDataModule(pl.LightningDataModule):
    def __init__(self, train_dataloaders, val_dataloader, train_num_queries, train_num_classes, val_num_queries,
                 val_num_classes, stage_period):
        super(REIDDataModule, self).__init__()
        self._train_dataloaders = train_dataloaders
        self._val_dataloader = val_dataloader
        self._train_num_queries = train_num_queries
        self._train_num_classes = train_num_classes
        self._val_num_queries = val_num_queries
        self._val_num_classes = val_num_classes
        self._stage_period = stage_period
        self._stage_idx = 0

    @property
    def train_num_queries(self):
        return self._train_num_queries

    @property
    def train_num_classes(self):
        return self._train_num_classes

    @property
    def val_num_queries(self):
        return self._val_num_queries

    @property
    def val_num_classes(self):
        return self._val_num_classes

    def train_dataloader(self):
        if self._stage_idx < len(self._stage_period):
            if self.trainer.current_epoch == self._stage_period[self._stage_idx]:
                print(
                    f"Current epoch approached {self._stage_period[self._stage_idx]}"
                    f"\nSwitching to the following transforms:\n {self._train_dataloaders[self._stage_idx + 1].dataset.transform}")
                self._stage_idx += 1
        train_dataloader = self._train_dataloaders[self._stage_idx]
        return train_dataloader

    def val_dataloader(self):
        return self._val_dataloader


def make_val_dataset(cfg, base_dataset):
    if cfg.DATASETS.VAL_NAMES is not None:
        CD_dataset = init_dataset(
            cfg.DATASETS.VAL_NAMES,
            root=cfg.DATASETS.VAL_ROOT,
            aug_per_image=cfg.SOLVER.AUG_PER_IMG
        )
    else:
        CD_dataset = base_dataset

    val_transforms = build_transforms_stage(cfg, is_train=False)
    val_set = ImageDataset(CD_dataset.query + CD_dataset.gallery, cfg.DATALOADER.COLOR_SPACE, transform=val_transforms)
    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=cfg.DATALOADER.NUM_WORKERS,
        collate_fn=test_collate_fn
    )
    return val_set, val_loader, len(CD_dataset.query), CD_dataset.num_train_pids


def make_data_loaders_with_stages(cfg):
    assert len(cfg.SOLVER.STAGE_TRANSFORMS) == len(cfg.SOLVER.STAGE_PERIOD), \
        f"STAGE_PERIOD and STATE_TRANSFORMS must have the same length, " \
        f"but got {cfg.SOLVER.STAGE_TRANSFORMS} and {cfg.SOLVER.STAGE_PERIOD} "
    train_loaders = []

    base_dataset = init_dataset(
        cfg.DATASETS.TRAIN_NAMES,
        root=cfg.DATASETS.TRAIN_ROOT,
        aug_per_image=cfg.SOLVER.AUG_PER_IMG
    )
    train_num_classes = base_dataset.num_train_pids
    val_set, val_loader, val_num_queries, val_num_classes = make_val_dataset(cfg, base_dataset)

    for st_idx in range(len(cfg.SOLVER.STAGE_TRANSFORMS) + 1):
        stage_transforms = cfg.SOLVER.STAGE_TRANSFORMS[:st_idx]
        train_transforms = build_transforms_stage(cfg, stage_transforms=stage_transforms, is_train=True)
        train_set = ImageDataset(base_dataset.train, cfg.DATALOADER.COLOR_SPACE, transform=train_transforms)
        if cfg.DATALOADER.SAMPLER == 'softmax':
            train_loader = DataLoader(
                train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=cfg.DATALOADER.NUM_WORKERS,
                collate_fn=train_collate_fn
            )
        else:
            train_loader = DataLoader(
                train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
                sampler=RandomIdentitySampler(base_dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE,
                                              fixed_epoch_steps=cfg.SOLVER.EVAL_INTERVAL),
                num_workers=cfg.DATALOADER.NUM_WORKERS, collate_fn=train_collate_fn
            )

        train_loaders.append(train_loader)

    return train_loaders, val_loader, len(base_dataset.query), train_num_classes, val_num_queries, val_num_classes


def make_pl_datamodule(cfg):
    return REIDDataModule(*make_data_loaders_with_stages(cfg), cfg.SOLVER.STAGE_PERIOD)
