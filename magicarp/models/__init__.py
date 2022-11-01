from abc import abstractclassmethod
from typing import Tuple, Any, Dict, Iterable, Callable
from dataclasses import dataclass

from torch import nn
import torch

from magicarp.data import DataElement
from magicarp.configs import ModelConfig

@dataclass
class ModelOutput:
    """
    Container for the output of a model. This is used to provide a consistent API for all models.
    """
    loss : torch.Tensor = None
    logits : torch.Tensor = None

class BaseCrossEncoder(nn.Module):
    def __init__(self, config : ModelConfig):
        super().___init__()
    
    @abstractclassmethod
    def  forward(self, input_A : DataElement, input_B : DataElement) -> ModelOutput:
        pass

class TextTextEncoder(BaseCrossEncoder):
    def __init__(self):
        super().___init__()
    
    def  forward(self, input_A : DataElement, input_B : DataElement) -> ModelOutput:
        pass

    
