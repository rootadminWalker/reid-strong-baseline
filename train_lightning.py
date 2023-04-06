#!/usr/bin/env python3
"""
@original_author:  sherlock
@contact: sherlockliao01@gmail.com
@pytorch_lightning_revise: rootadminWalker
"""

import os

import pytorch_lightning as pl
from lightning_fabric.utilities.seed import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, RichProgressBar
from pytorch_lightning.loggers import TensorBoardLogger

from data import make_pl_datamodule
from engine.reid_module import PersonReidModule
from solver import build_warmup_lr, build_direct_set_lr, build_fine_tune
from utils import setup_cli


def train(cfg):
    if cfg.RANDOM_SEED is not None:
        seed_everything(cfg.RANDOM_SEED)

    output_dir = cfg.OUTPUT_DIR
    tb_logs_path = cfg.TB_LOG_DIR

    # prepare dataset
    datamodule = make_pl_datamodule(cfg)
    # datamodule = NewREIDDataModule(cfg)

    # Setup callbacks
    warmup_lr = build_warmup_lr(cfg)
    direct_set_lr = build_direct_set_lr(cfg)
    lr_monitor = LearningRateMonitor(
        logging_interval='step'
    )
    epoch_checkpoint_cb = ModelCheckpoint(
        dirpath=output_dir,
        every_n_epochs=cfg.SOLVER.CHECKPOINT_PERIOD,
        save_top_k=-1
    )
    top_mAP_checkpoint_cb = ModelCheckpoint(
        dirpath=output_dir,
        save_top_k=1,
        monitor='mAP',
        mode='max',
        filename=cfg.MODEL.NAME + "_best-mAP-{epoch:2d}-{mAP}"
    )
    top_rank1_checkpoint_cb = ModelCheckpoint(
        dirpath=output_dir,
        save_top_k=1,
        monitor='rank1',
        mode='max',
        filename=cfg.MODEL.NAME + "_best-rank1-{epoch:2d}-{rank1}"
    )

    logger = TensorBoardLogger(
        os.path.join(output_dir, tb_logs_path),
        name="reid-train",
        version=0
    )

    callbacks = [
        warmup_lr,
        direct_set_lr,
        lr_monitor,
        epoch_checkpoint_cb,
        top_mAP_checkpoint_cb,
        top_rank1_checkpoint_cb,
        RichProgressBar()
    ]
    if cfg.SOLVER.UNFREEZE_AT_EPOCH is not None:
        callbacks.append(build_fine_tune(cfg))

    # No center not supported now
    assert cfg.MODEL.IF_WITH_CENTER == 'yes', "Not supported for no center"

    # Add for using self trained model
    assert cfg.MODEL.PRETRAIN_CHOICE in ['self', 'imagenet'], \
        'Only support pretrain_choice for imagenet and self, but got {}'.format(cfg.MODEL.PRETRAIN_CHOICE)
    checkpoint_path = None
    if cfg.MODEL.PRETRAIN_CHOICE == 'self':
        checkpoint_path = cfg.MODEL.PRETRAIN_PATH

    module = PersonReidModule(
        cfg=cfg,
        train_num_classes=datamodule.train_num_classes,
        val_num_queries=datamodule.val_num_queries
    )
    trainer = pl.Trainer(
        accelerator=cfg.MODEL.DEVICE,
        devices=list(map(int, cfg.MODEL.DEVICE_ID)),
        benchmark=True,
        logger=logger,
        callbacks=callbacks,
        max_epochs=cfg.SOLVER.MAX_EPOCHS,
        val_check_interval=cfg.SOLVER.EVAL_INTERVAL,
        num_sanity_val_steps=0,
        strategy=cfg.SOLVER.STRATEGY,
        num_nodes=cfg.SOLVER.NUM_NODES,
        reload_dataloaders_every_n_epochs=1
    )
    trainer.fit(module, datamodule=datamodule, ckpt_path=checkpoint_path)


def main():
    cfg, args = setup_cli()
    print(cfg)
    train(cfg)


if __name__ == '__main__':
    main()
