from typing import Dict

import wandb

from torch import nn
import functools

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
    do_log = config.log_interval > 0
    do_save = config.save_interval > 0
    do_val = config.val_interval > 0
    return {
        "log" : do_log and steps % config.log_interval == 0,
        "save" : do_save and steps % config.save_interval == 0,
        "val" : do_val and steps % config.val_interval == 0
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

    def __init__(self, model : 'CrossEncoder', loss_fn : 'CrossEncoderLoss'):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn

    def __forward__(self, **kwargs):
        return self.loss_fn(self.model(**kwargs)).loss

# Layer freezing 
# Reference: https://github.com/Dahoas/reward-modeling/blob/main/reward-modeling/utils.py

def rhasattr(obj, attr):
    """A chain-able attribute version of hasattr. For example, to check if
    `obj` has the attribute `foo.bar.baz`, you can use:
        `rhasattr(obj, "foo.bar.baz")`
    Reference: https://stackoverflow.com/a/67303315
    """
    _nested_attrs = attr.split(".")
    _curr_obj = obj
    for _a in _nested_attrs[:-1]:
        if hasattr(_curr_obj, _a):
            _curr_obj = getattr(_curr_obj, _a)
        else:
            return False
    return hasattr(_curr_obj, _nested_attrs[-1])


def rgetattr(obj, attr: str, *args):
    """A chain-able attribute version of getattr. For example, to get the
    attribute `foo.bar.baz` from `obj`, you can use:
        `rgetattr(obj, "foo.bar.baz")`
    Reference: https://stackoverflow.com/a/31174427
    """

    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))


def findattr(obj, attrs):
    for attr in attrs:
        if rhasattr(obj, attr):
            return rgetattr(obj, attr)


def hf_get_causal_hidden_layers(model: nn.Module):
    """Returns the hidden layers of the specified model.
    NOTE: Different model configurations have different hidden layer attribute names.
        - transformer.h: (BloomForCausalLM, GPT2LMHeadModel, GPTJForCausalLM)
        - model.decoder.layers: (OPTForCausalLM)
        - gpt_neox.layers: (GPTNeoXForCausalLM)
    """
    hidden_layers_attrs = [
        "transformer.h",
        "model.decoder.layers",
        "encoder.layer",
        "gpt_neox.layers",
    ]
    return findattr(model, hidden_layers_attrs)


def freeze_bottom_causal_layers(model: nn.Module, num_layers_unfrozen):
    """Freezes the bottom transformer block layers of the specified model."""
    hidden_layers = hf_get_causal_hidden_layers(model)
    num_layers_unfrozen = int(len(hidden_layers) * num_layers_unfrozen) if type(num_layers_unfrozen) is float else num_layers_unfrozen
    if num_layers_unfrozen == 0:
        hidden_layers_to_freeze = list(hidden_layers)
    elif num_layers_unfrozen > 0:
        hidden_layers_to_freeze = list(hidden_layers)[:-num_layers_unfrozen]
    else:
        hidden_layers_to_freeze = []
    for layer in hidden_layers_to_freeze:
        layer.requires_grad_(False)
