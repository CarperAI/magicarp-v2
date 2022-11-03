from dataclasses import dataclass
from typing import Dict, Any
from torchtyping import TensorType

import torch
from torch import Tensor 

@dataclass
class DataElement:
    """
    Container for any kind of data that we may wish to pass to an encoder or model.
    In subclasses, all attributes are expected to be tensortypes, with the sole exception
    of nondatafields. This dictionary can be used to store arbitrary data that isn't
    required for the model. Use this to store things like flags, untokenized text, or
    tags for the data.

    :param nondatafields: Dictionary of arbitrary data that isn't required for the model.
    :type nondatafields: Dict[str, Any]
    """

    nondatafields : Dict[str, Any] = None

    def to(self, device : torch.device) -> 'DataElement':
        """
        Move all tensor attributes to a given device.

        :param device: Device to move tensors to.
        :type device: torch.device
        """

        for k, v in self.__dict__.items():
            if isinstance(v, Tensor):
                setattr(self, k, v.to(device))

        return self
    
    def to_dict(self) -> Dict[str, Tensor]:
        """
        Convert all tensor attributes to a dictionary.

        :return: Dictionary of tensor attributes.
        :rtype: Dict[str, Tensor]
        """

        return {k : v for k, v in self.__dict__.items() if isinstance(v, Tensor)}
        
@dataclass
class TextElement(DataElement):
    """
    Element for any data originating as text. Used to provide input to language model.
    
    :param input_ids: Tensor of token indices for the input text.
    :type input_ids: torch.Tensor

    :param attention_mask: Tensor of attention mask over the input text.
    :type attention_mask: torch.Tensor
    """

    input_ids : TensorType["batch_size", "seq_len"] = None
    attention_mask : TensorType["batch_size", "seq_len"] = None

@dataclass
class ImageElement(DataElement):
    """
    Element for any data originating as an image. Used to provide input to vision model.
    
    :param pixel_values: Tensor of pixel values for the input image.
    :type pixel_values: torch.Tensor
    """

    pixel_values : TensorType["batch_size", "channels", "height", "width"] = None

@dataclass
class AudioElement:
    """
    Element for any data originating as audio. Used to provide input to audio model.
    
    :param waveform: Tensor of audio samples for the input audio.
    :type waveform: torch.Tensor
    """

    waveform : TensorType["batch", "samples"] = None