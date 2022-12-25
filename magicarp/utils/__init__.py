from typing import Dict

import wandb

from torch import nn

from magicarp.configs import TrainConfig
from magicarp.models import CrossEncoder
from magicarp.losses import CrossEncoderLoss

def get_intervals(config : TrainConfig, steps : int) -> Dict[str, bool]:
    """
    Creates a dictionary specifiying whether or not certain tasks should be done on the given step (i.e. logging/saving/eval)

    :param config: The config expected to contain intervals for tasks to be done.
    :type config: TrainConfig

    :param steps: The number of steps that have been taken.
    :type steps: int

    :return: A dictionary containing the intervals for each task.
    :rtype: Dict[str, bool]
    """
    return {
        "log" : steps % config.log_interval == 0,
        "save" : steps % config.save_interval == 0,
        "val" : steps % config.val_interval == 0
    }

def wandb_start(config : TrainConfig):
    """
    Starts wandb logging with the given config.

    :param config: The config to use for wandb.
    :type config: TrainConfig
    """
    wandb.init(project=config.wandb_project, entity=config.wandb_entity, config=config.__dict__)

class AttachLoss(nn.Module):
    """
        Attach a model to a loss function to produce a combined model
        that directly outputs loss. Currenty only works with EndLoss.  
    """

    def __init__(self, model : CrossEncoder, loss_fn : CrossEncoderLoss):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn

    def __forward__(self, **kwargs):
        return self.loss_fn(self.model(**kwargs)).loss