from typing import Dict

import wandb

from magicarp.configs import TrainConfig

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
