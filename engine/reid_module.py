import logging

import pytorch_lightning as pl

from config import cfg
from data import make_data_loader
from layers import make_loss_with_center
from modeling import build_model
from solver import make_optimizer_with_center
from solver.build import build_scheduler
from utils.reid_metric import R1_mAP


class PersonReidModule(pl.LightningModule):
    def __init__(self, cfg, num_classes, num_queries):
        super(PersonReidModule, self).__init__()
        self.automatic_optimization = False

        self.save_hyperparameters()
        self.cfg = cfg

        self.model = build_model(cfg, num_classes)
        self.loss, self.center_criterion = make_loss_with_center(cfg, num_classes)
        self.metric = R1_mAP(num_queries, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
        self.console = self.__setup_console_logging()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        opt, opt_center = make_optimizer_with_center(self.cfg, self.model, self.center_criterion)
        multistep_scheduler = build_scheduler(opt, self.cfg)
        return [opt, opt_center], multistep_scheduler

    def on_train_epoch_start(self):
        self.metric.reset()

    def training_step(self, batch, batch_idx):
        opt, opt_center = self.optimizers()
        opt.zero_grad()
        opt_center.zero_grad()

        img, target = batch
        # print(batch_idx, img.shape)
        score, global_feat = self.model.train_forward(img)
        losses = self.loss(score, global_feat, target)
        total_train_loss, train_loss_components = losses

        self.manual_backward(total_train_loss)
        opt.step()
        for param in self.center_criterion.parameters():
            param.grad.data *= (1. / self.cfg.SOLVER.CENTER_LOSS_WEIGHT)
        opt_center.step()

        train_acc = (score.max(1)[1] == target).float().mean()

        train_output = {"loss": total_train_loss, "train_acc": train_acc,
                        "train_loss_components": train_loss_components}
        self.__log_losses(train_output, mode='train')
        return train_output

    # def on_train_batch_end(self, output, batch, batch_idx):
    #     print("oweifjewiofwe jfoiwejf wepjf w jwejfjwoif jwpf iowjf iowjf w")

    def validation_step(self, batch, batch_idx):
        # raise NotImplementedError()
        data, pids, camids = batch
        feat = self.model(data)
        self.metric.update([feat, pids, camids])

    def on_validation_epoch_end(self):
        cmc, mAP = self.metric.compute()
        mAP = round(mAP * 100, 1)

        self.console.info(f"Validation Results - Epoch: {self.current_epoch}")
        self.console.info(f"mAP: {mAP}%")
        self.log('mAP', mAP)
        for r in [1, 5, 10]:
            r_cmc = round(cmc[r - 1] * 100, 1)
            self.console.info(f"CMC curve, Rank-{r:<3}:{r_cmc}%")
            self.log(f'rank{r}', r_cmc)

    def on_train_epoch_end(self):
        self.lr_schedulers().step()
        # print(self.optimizers()[0].param_groups[0]['lr'])
        # print(self.lr_schedulers().last_epoch, self.lr_schedulers().milestones)

    @staticmethod
    def __setup_console_logging():
        logging.getLogger("pytorch_lightning").setLevel(logging.INFO)
        console = logging.getLogger("pytorch_lightning.core")
        return console

    def __log_losses(self, loss_output, mode='train'):
        self.log(f'{mode}_loss', loss_output['loss'])
        self.log(f'{mode}_acc', loss_output[f'{mode}_acc'])
        for loss_name, loss_val in loss_output[f'{mode}_loss_components'].items():
            self.log(f'{mode}_{loss_name}', loss_val)


if __name__ == '__main__':
    cfg.merge_from_file("../configs/am-softmax_triplet_with_center.yml")
    cfg.DATASETS.ROOT_DIR = "/media/root_walker/DATA/datasets/"
    cfg.freeze()
    train_loader, val_loader, num_query, num_classes = make_data_loader(cfg)
    module = PersonReidModule(cfg, num_query, num_query)
