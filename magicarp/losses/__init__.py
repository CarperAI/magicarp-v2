from abc import abstractmethod
from typing import Optional, Any, Dict
from torchtyping import TensorType

import sys
from torch import nn

from magicarp.data import DataElement
from magicarp.configs import TrainConfig
from magicarp.models import ModelOutput

# Registry for all losses
_LOSSES: Dict[str, Any] = {}

def register_loss(name):
    """Decorator used to register a loss function
    Args:
        name: Name of the loss
    """

    def register_class(cls, name):
        _LOSSES[name] = cls
        setattr(sys.modules[__name__], name, cls)
        return cls

    if isinstance(name, str):
        name = name.lower()
        return lambda c: register_class(c, name)

    cls = name
    name = cls.__name__
    register_class(cls, name.lower())

    return cls

def get_loss(name):
    """Get a loss function by name
    Args:
        name: Name of the loss
    """
    return _LOSSES[name.lower()]

class CrossEncoderLoss(nn.Module):
    """
        Abstract base for all cross-encoder (reward model/reranker) losses.

        :param config: TrainConfig for the loss
        :type config: TrainConfig
    """
    def __init__(self, config : TrainConfig):
        super().__init__()
        self.query = config.query_modality
        
class EndLoss(CrossEncoderLoss):
    """
        Abstract class for losses where scores are given at end of sequence.
    """
    
    @abstractmethod
    def forward(
        self,
        model_out : ModelOutput,
        expected_scores : Optional[TensorType["batch"]] = None
    ) -> ModelOutput:
        """
            Compute loss given scores. Some losses may require expected scores, while others are entirely unsupervised.

            :param model_out: ModelOutput with scores
            :type model_out: ModelOutput

            :param expected_scores: Expected scores, defaults to None
            :type expected_scores: Optional[TensorType["batch"]]

            :return: ModelOutput with loss and scores from model
            :rtype: ModelOutput
        """
        pass

class StepLoss(CrossEncoderLoss):
    """
        Abstract class for losses that are computed at each step in the sequence.
    """

    @abstractmethod
    def forward(
        self,
        model_out : ModelOutput,
        input_sequence : Optional[DataElement] = None,
        expected_scores : Optional[TensorType["batch", "seq"]] = None
    ) -> ModelOutput:
        """
            Compute loss given scores. Some losses may require expected scores, while others are entirely unsupervised.
            Optionally takes input sequences to compute loss.

            :param model_output: ModelOutput with scores
            :type model_output: ModelOutput

            :param input_sequence: Input sequence, defaults to None
            :type input_sequence: Optional[DataElement], optional

            :param expected_scores: Expected scores, defaults to None
            :type expected_scores: Optional[TensorType["batch", "seq"]]

            :return: ModelOutput with loss and scores from model
            :rtype: ModelOutput
        """
        pass