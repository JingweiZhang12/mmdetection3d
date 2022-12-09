from mmdet3d.registry import HOOKS
from mmdet3d.datasets.transforms import ObjectSample

from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmengine.runner import Runner


@HOOKS.register_module()
class DisableObjectSampleHook(Hook):
    """The hook of disabling augmentations during training.

    Args:
        num_last_epochs (int): The number of latter epochs in the end of the
            training to close the data augmentation. Default: 15.
       skip_type_keys (list[str], optional): Sequence of type string to be
            skipped in the data pipeline. Default: ('ObjectSample')
    """

    def __init__(self, disable_after_epoch: int = 15):
        self.disable_after_epoch = disable_after_epoch
        self._restart_dataloader = False

    def before_train_epoch(self, runner: Runner):
        """Close augmentation.

        Args:
            runner (Runner): The runner.
        """
        epoch = runner.epoch
        train_loader = runner.train_dataloader
        model = runner.model
        # TODO: refactor after mmengine using model wrapper
        if is_model_wrapper(model):
            model = model.module
        if epoch == self.disable_after_epoch:
            runner.logger.info('Disable ObjectSample')
            for transform in runner.train_dataloader.dataset.pipeline.transforms:
                if isinstance(transform, ObjectSample):
                    assert hasattr(transform, 'disabled')
                    transform.disabled = True
            # The dataset pipeline cannot be updated when persistent_workers
            # is True, so we need to force the dataloader's multi-process
            # restart. This is a very hacky approach.
            if hasattr(train_loader, 'persistent_workers'
                       ) and train_loader.persistent_workers is True:
                train_loader._DataLoader__initialized = False
                train_loader._iterator = None
                self._restart_dataloader = True
        else:
            # Once the restart is complete, we need to restore
            # the initialization flag.
            if self._restart_dataloader:
                train_loader._DataLoader__initialized = True
