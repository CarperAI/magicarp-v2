from abc import abstractclassmethod
from typing import Tuple, Any, Dict, Iterable, Callable, Optional
from torchtyping import TensorType
from dataclasses import dataclass

from torch import nn
import torch
from transformers import AutoConfig

from magicarp.data import DataElement
from magicarp.configs import ModelConfig

@dataclass
class ModelOutput:
    """
    Container for the output of a model. This is used to provide a consistent API for all models.
    """
    loss : torch.tensor = None # Loss across batch: 0d tensor
    scores : TensorType["batch"] = None

class CrossEncoder(nn.Module):
    """
    Base class for any CrossEncoder model.
    """
    
    def __init__(self, config : ModelConfig):
        super().__init__()

        tf_cfg = AutoConfig.from_pretrained(config.model_path)
        self.score_head = nn.Linear(tf_cfg.hidden_size, 1)
    
    @abstractclassmethod
    def preprocess(self, input_A : Iterable[Any], input_B : Iterable[Any]) -> Any:
        """
        Preprocesses the input data into a format that can be fed into the model. Normally calls upon a feature extractor or a tokenizer.
        
        :param input_A: An iterable of the first input data in its raw form.
        :type input_A: Iterable[Any]

        :param input_B: An iterable of the second input data in its raw form.
        :type input_B: Iterable[Any]
    
        :return: The preprocessed data. Could be a single item, a tuple, or an iterable.
        :rtype: Any
        """
        pass

    @abstractclassmethod
    def  forward(self, x : Iterable[DataElement], scores : Optional[TensorType["batch"]] = None) -> ModelOutput:
        """
        Forward call for CrossEncoder. Takes one or more DataElements and returns a ModelOutput

        :param x: One or more DataElements
        :type x: Iterable[DataElement]

        :param scores: Scores for pairwise relevance. If provided, loss is computed.
        :type scores: Optional[TensorType["batch"]]

        :return: Output of model, consisting of loss (if scores provided) and scores.
        :rtype: ModelOutput
        """
        pass

