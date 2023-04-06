from pytorch_lightning.callbacks import BaseFinetuning


class FineTuning(BaseFinetuning):
    def __init__(self, unfreeze_at_epoch=10):
        super(FineTuning, self).__init__()
        self._unfreeze_at_epoch = unfreeze_at_epoch

    def freeze_before_training(self, pl_module) -> None:
        self.freeze(pl_module.model.base)

    def finetune_function(
            self, pl_module, current_epoch, optimizer, opt_idx
    ) -> None:
        if current_epoch == self._unfreeze_at_epoch and opt_idx == 0:
            self.unfreeze_and_add_param_group(
                modules=pl_module.model.base,
                optimizer=optimizer,
                train_bn=True)
