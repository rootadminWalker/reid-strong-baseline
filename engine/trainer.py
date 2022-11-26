# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import logging
import os

import torch
import torch.nn as nn
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import RunningAverage

from data import make_data_loader
from modeling import build_model
from utils.reid_metric import R1_mAP
from pytorch_lightning import LightningModule

global ITER
ITER = 0
best_mAP = 0
best_rank1 = 0


class TrainModule(LightningModule):
    def __init__(self, cfg, num_classes):
        super(TrainModule, self).__init__()
        self.model = build_model(cfg, num_classes)


def create_supervised_trainer(model, optimizer, loss_fn,
                              device=None):
    """
    Factory function for creating a trainer for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        optimizer (`torch.optim.Optimizer`): the optimizer to use
        loss_fn (torch.nn loss function): the loss function to use
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.

    Returns:
        Engine: a trainer engine with supervised update function
    """
    if device:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)

    def _update(engine, batch):
        loss_components = []
        model.train()
        optimizer.zero_grad()
        img, target = batch
        img = img.to(device) if torch.cuda.device_count() >= 1 else img
        target = target.to(device) if torch.cuda.device_count() >= 1 else target
        score, feat = model(img)
        losses = loss_fn(score, feat, target)
        if isinstance(losses, tuple):
            total_loss, *loss_components = losses
        else:
            total_loss = losses

        total_loss.backward()
        optimizer.step()
        # compute acc
        acc = (score.max(1)[1] == target).float().mean()
        return {'total_loss': total_loss.item(), 'acc': acc.item(), 'loss_components': loss_components}

    return Engine(_update)


def create_supervised_trainer_with_center(model, center_criterion, optimizer, optimizer_center, loss_fn,
                                          cetner_loss_weight,
                                          device=None):
    """
    Factory function for creating a trainer for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        optimizer (`torch.optim.Optimizer`): the optimizer to use
        loss_fn (torch.nn loss function): the loss function to use
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.

    Returns:
        Engine: a trainer engine with supervised update function
    """
    if device:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)

    def _update(engine, batch):
        loss_components = []
        model.train()
        optimizer.zero_grad()
        optimizer_center.zero_grad()
        img, target = batch
        img = img.to(device) if torch.cuda.device_count() >= 1 else img
        target = target.to(device) if torch.cuda.device_count() >= 1 else target
        score, feat = model(img)
        losses = loss_fn(score, feat, target)
        if isinstance(losses, tuple):
            total_loss, *loss_components = losses
        else:
            total_loss = losses
        # print("Total loss is {}, center loss is {}".format(loss, center_criterion(feat, target)))
        total_loss.backward()
        optimizer.step()
        for param in center_criterion.parameters():
            param.grad.data *= (1. / cetner_loss_weight)
        optimizer_center.step()

        # compute acc
        acc = (score.max(1)[1] == target).float().mean()
        return {'total_loss': total_loss.item(), 'acc': acc.item(), 'loss_components': loss_components}

    return Engine(_update)


def create_supervised_evaluator(model, metrics,
                                device=None):
    """
    Factory function for creating an evaluator for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        metrics (dict of str - :class:`ignite.metrics.Metric`): a map of metric names to Metrics
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.
    Returns:
        Engine: an evaluator engine with supervised inference function
    """
    if device:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            data, pids, camids = batch
            data = data.to(device) if torch.cuda.device_count() >= 1 else data
            feat = model(data)
            return feat, pids, camids

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


def do_train(
        cfg,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        loss_fn,
        num_query,
        start_epoch
):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    output_dir = cfg.OUTPUT_DIR
    device = cfg.MODEL.DEVICE
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("reid_baseline")
    logger.info("Start training")
    trainer = create_supervised_trainer(model, optimizer, loss_fn, device=device)
    evaluator = create_supervised_evaluator(model, metrics={
        'r1_mAP': R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)}, device=device)
    checkpointer = ModelCheckpoint(output_dir, cfg.MODEL.NAME, n_saved=10, require_empty=False)
    timer = Timer(average=True)

    trainer.add_event_handler(Events.EPOCH_COMPLETED(every=checkpoint_period), checkpointer, {'model': model,
                                                                                              'optimizer': optimizer})
    timer.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

    # average metric to attach on trainer
    RunningAverage(output_transform=lambda x: x['total_loss']).attach(trainer, 'avg_total_loss')
    RunningAverage(output_transform=lambda x: x['acc']).attach(trainer, 'avg_acc')
    if cfg.DATALOADER.SAMPLER == 'softmax_triplet':
        RunningAverage(output_transform=lambda x: x['loss_components'][0]).attach(trainer, 'avg_id_loss')
        RunningAverage(output_transform=lambda x: x['loss_components'][1]).attach(trainer, 'avg_triplet_loss')

    @trainer.on(Events.STARTED)
    def start_training(engine):
        engine.state.epoch = start_epoch

    @trainer.on(Events.EPOCH_STARTED)
    def adjust_learning_rate(engine):
        scheduler.step()

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        global ITER
        ITER += 1

        if ITER % log_period == 0:
            id_triplet_text = ""
            if cfg.DATALOADER.SAMPLER == 'softmax_triplet':
                id_triplet_text = f"ID Loss: {engine.state.metrics['avg_id_loss']:.3f}, Triplet Loss: {engine.state.metrics['avg_triplet_loss']:.3f}, "
            logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, {}Acc: {:.3f}, Base Lr: {:.2e}"
                        .format(engine.state.epoch, ITER, len(train_loader),
                                engine.state.metrics['avg_total_loss'], id_triplet_text,
                                engine.state.metrics['avg_acc'],
                                scheduler.get_lr()[0]))
        if len(train_loader) == ITER:
            ITER = 0

    # adding handlers using `trainer.on` decorator API
    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        logger.info('Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]'
                    .format(engine.state.epoch, timer.value() * timer.step_count,
                            train_loader.batch_size / timer.value()))
        logger.info('-' * 10)
        timer.reset()

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        if engine.state.epoch % eval_period == 0:
            evaluator.run(val_loader)
            cmc, mAP = evaluator.state.metrics['r1_mAP']
            logger.info("Validation Results - Epoch: {}".format(engine.state.epoch))
            logger.info("mAP: {:.1%}".format(mAP))
            for r in [1, 5, 10]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

    trainer.run(train_loader, max_epochs=epochs)


def do_train_with_center(
        cfg,
        model,
        center_criterion,
        train_loader,
        val_loader,
        optimizer,
        optimizer_center,
        scheduler,
        loss_fn,
        num_query,
        start_epoch
):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    output_dir = cfg.OUTPUT_DIR
    device = cfg.MODEL.DEVICE
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("reid_baseline")
    logger.info("Start training")
    trainer = create_supervised_trainer_with_center(model, center_criterion, optimizer, optimizer_center, loss_fn,
                                                    cfg.SOLVER.CENTER_LOSS_WEIGHT, device=device)
    evaluator = create_supervised_evaluator(model, metrics={
        'r1_mAP': R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)}, device=device)
    checkpointer = ModelCheckpoint(output_dir, cfg.MODEL.NAME, n_saved=10, require_empty=False)
    tb_logger = TensorboardLogger(log_dir=cfg.TB_LOG_DIR)
    timer = Timer(average=True)

    trainer.add_event_handler(Events.EPOCH_COMPLETED(every=checkpoint_period), checkpointer, {'model': model,
                                                                                              'optimizer': optimizer,
                                                                                              'center_param': center_criterion,
                                                                                              'optimizer_center': optimizer_center})

    timer.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

    # average metric to attach on trainer
    RunningAverage(output_transform=lambda x: x['total_loss']).attach(trainer, 'avg_total_loss')
    RunningAverage(output_transform=lambda x: x['acc']).attach(trainer, 'avg_acc')
    RunningAverage(output_transform=lambda x: x['loss_components'][0]).attach(trainer, 'avg_id_loss')
    RunningAverage(output_transform=lambda x: x['loss_components'][1]).attach(trainer, 'avg_center_loss')
    if 'triplet_center' in cfg.MODEL.METRIC_LOSS_TYPE:
        RunningAverage(output_transform=lambda x: x['loss_components'][2]).attach(trainer, 'avg_triplet_loss')
    if 'CTL' in cfg.MODEL.METRIC_LOSS_TYPE:
        RunningAverage(output_transform=lambda x: x['loss_components'][3]).attach(trainer, 'avg_ctl_loss')

    tb_logger.attach_output_handler(
        trainer,
        tag="training",
        metric_names=['avg_total_loss', 'avg_acc', 'avg_id_loss', 'avg_center_loss', 'avg_triplet_loss'],
        event_name=Events.ITERATION_COMPLETED
    )

    def save_best_checkpoint(epoch, best_metric_name, epoch_test_results):
        torch.save({
            "model": model.state_dict(),
            "epoch": epoch,
            "epoch_test_results": epoch_test_results
        }, os.path.join(cfg.OUTPUT_DIR, f"{cfg.MODEL.NAME}_best_{best_metric_name}.pt"))

    @trainer.on(Events.STARTED)
    def start_training(engine):
        engine.state.epoch = start_epoch

    @trainer.on(Events.EPOCH_STARTED)
    def adjust_learning_rate(engine):
        scheduler.step()

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        global ITER
        ITER += 1

        if ITER % log_period == 0:
            triplet_text = ''
            ctl_text = ''
            if 'triplet_center' in cfg.MODEL.METRIC_LOSS_TYPE:
                triplet_text = f"Triplet Loss: {engine.state.metrics['avg_triplet_loss']:.3f}, "
            if 'CTL' in cfg.MODEL.METRIC_LOSS_TYPE:
                ctl_text = f"CTL Loss: {engine.state.metrics['avg_ctl_loss']:.3f}, "

            logger.info(
                "Epoch[{}] Iteration[{}/{}] Total Loss: {:.3f}, ID Loss: {:.3f}, Center Loss: {:.3f}, {}{}Acc: {:.3f}, Base Lr: {:.2e}"
                .format(engine.state.epoch, ITER, len(train_loader),
                        engine.state.metrics['avg_total_loss'], engine.state.metrics['avg_id_loss'],
                        engine.state.metrics['avg_center_loss'], triplet_text, ctl_text,
                        engine.state.metrics['avg_acc'],
                        scheduler.get_lr()[0]))
        if len(train_loader) == ITER:
            ITER = 0

    # adding handlers using `trainer.on` decorator API
    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        logger.info('Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]'
                    .format(engine.state.epoch, timer.value() * timer.step_count,
                            train_loader.batch_size / timer.value()))
        logger.info('-' * 10)
        timer.reset()

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        global best_mAP, best_rank1
        if engine.state.epoch % eval_period == 0:
            evaluator.run(val_loader)
            cmc, mAP = evaluator.state.metrics['r1_mAP']
            logger.info("Validation Results - Epoch: {}".format(engine.state.epoch))
            logger.info("mAP: {:.1%}".format(mAP))
            for r in [1, 5, 10]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

            epoch_test_result = {
                'mAP': mAP,
                'rank1': cmc[0],
                'rank5': cmc[4],
                'rank10': cmc[9]
            }
            if mAP > best_mAP:
                best_mAP = mAP
                logger.info("Current epoch has the best mAP score, saving checkpoint")
                save_best_checkpoint(engine.state.epoch, 'mAP', epoch_test_result)
            if cmc[0] > best_rank1:
                best_rank1 = cmc[0]
                logger.info("Current epoch has the best rank1 score, saving checkpoint")
                save_best_checkpoint(engine.state.epoch, 'rank1', epoch_test_result)
            else:
                logging.info('No imporvement this epoch, not saving anything')
            logger.info('-' * 10)

    trainer.run(train_loader, max_epochs=epochs)
