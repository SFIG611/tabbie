
from overrides import overrides
from typing import Dict, Any
from allennlp.training import EpochCallback


@EpochCallback.register("write_callback")
class WriteCallback(EpochCallback):
    """
    An optional callback that you can pass to the `GradientDescentTrainer` that will be called at
    the end of every epoch (and before the start of training, with `epoch=-1`). The default
    implementation does nothing. You can implement your own callback and do whatever you want, such
    as additional modifications of the trainer's state in between epochs.
    """

    @overrides
    def __call__(
        self,
        trainer: "GradientDescentTrainer",
        metrics: Dict[str, Any],
        epoch: int,
        is_master: bool,
    ) -> None:
        super(EpochCallback, self).__init__()
        print('epoch: {}'.format(epoch))
        print(trainer.cuda_device)
        if epoch == 0:
            trainer.model.cache_util.close()


