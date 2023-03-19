#!/usr/bin/env python3
"""
@original_author:  sherlock
@contact: sherlockliao01@gmail.com
@pytorch_lightning_revise: rootadminWalker
"""

import os
import sys

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, RichProgressBar
from pytorch_lightning.loggers import TensorBoardLogger

sys.path.append('.')
from engine.reid_module import PersonReidModule
from solver.build import build_warmup_lr, build_direct_set_lr

from utils import setup_cli, setup_loggers
from data import make_pl_datamodule


def train(cfg):
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

    # No center not supported now
    assert cfg.MODEL.IF_WITH_CENTER == 'yes', "Not supported for no center"

    # Add for using self trained model
    assert cfg.MODEL.PRETRAIN_CHOICE in ['self', 'imagenet'], \
        'Only support pretrain_choice for imagenet and self, but got {}'.format(cfg.MODEL.PRETRAIN_CHOICE)
    checkpoint_path = None
    if cfg.MODEL.PRETRAIN_CHOICE == 'self':
        checkpoint_path = cfg.MODEL.PRETRAIN_PATH

    module = PersonReidModule(cfg, num_classes=datamodule.num_classes, num_queries=datamodule.num_queries)
    trainer = pl.Trainer(
        accelerator=cfg.MODEL.DEVICE,
        devices=list(map(int, cfg.MODEL.DEVICE_ID)),
        benchmark=True,
        logger=logger,
        callbacks=[
            warmup_lr,
            direct_set_lr,
            lr_monitor,
            epoch_checkpoint_cb,
            top_mAP_checkpoint_cb,
            top_rank1_checkpoint_cb,
            RichProgressBar()
        ],
        max_epochs=cfg.SOLVER.MAX_EPOCHS,
        # check_val_every_n_epoch=cfg.SOLVER.EVAL_PERIOD,
        val_check_interval=cfg.SOLVER.EVAL_INTERVAL,
        num_sanity_val_steps=0,
    )
    trainer.fit(module, datamodule=datamodule, ckpt_path=checkpoint_path)


def main():
    cfg, args = setup_cli()
    logger = setup_loggers(args, cfg.OUTPUT_DIR)
    train(cfg)


if __name__ == '__main__':
    main()
