from abc import abstractclassmethod
from typing import Tuple, Any, Dict, Iterable, Callable
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
    loss : torch.Tensor = None
    scores : torch.Tensor = None

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
        """
        pass

    @abstractclassmethod
    def  forward(self, x : Iterable[DataElement]) -> ModelOutput:
        pass

