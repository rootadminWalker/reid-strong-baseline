# encoding: utf-8
"""
@original_author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from pytorch_lightning import Callback


class WarmupLR(Callback):
    def __init__(self, base_lr, warmup_factor, warmup_iters, warmup_method):
        self.base_lr = base_lr
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method

        if self.warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(self.warmup_method)
            )

    @staticmethod
    def __change_lr(optimizer, new_lr):
        for pg in optimizer.param_groups:
            pg["lr"] = new_lr

    def on_train_epoch_start(self, trainer, pl_module):
        optimizer = trainer.strategy.optimizers[0]
        current_epoch = trainer.current_epoch + 1
        warmup_factor = 1
        if current_epoch <= self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = current_epoch / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha

            self.__change_lr(optimizer, self.base_lr * warmup_factor)


class DirectSetLR(Callback):
    def __init__(self, direct_steps, direct_lrs):
        self.direct_steps = direct_steps
        self.direct_lrs = direct_lrs
        self._direct_idx = 0
        assert len(self.direct_steps) == len(self.direct_lrs), "Two must be the same"

    @staticmethod
    def __change_lr(optimizer, new_lr):
        for pg in optimizer.param_groups:
            pg["lr"] = new_lr

    def on_train_epoch_start(self, trainer, pl_module):
        optimizer = trainer.strategy.optimizers[0]
        current_epoch = trainer.current_epoch
        if self._direct_idx < len(self.direct_steps):
            if current_epoch == self.direct_steps[self._direct_idx]:
                self.__change_lr(optimizer, self.direct_lrs[self._direct_idx])
                self._direct_idx += 1
