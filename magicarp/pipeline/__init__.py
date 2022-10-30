from abc import abstractclassmethod
from typing import Tuple, Any, Dict, Iterable, Callable

import sys

from torch.utils.data import Dataset, DataLoader
from torch import Tensor
import torch

from magicarp.data import DataElement

_DATAPIPELINE: Dict[str, any] = {}  # registry of all pipelines

def register_datapipeline(name):
    """Decorator used register a CARP architecture
    Args:
        name: Name of the architecture
    """

    def register_class(cls, name):
        _DATAPIPELINE[name] = cls
        setattr(sys.modules[__name__], name, cls)
        return cls

    if isinstance(name, str):
        name = name.lower()
        return lambda c: register_class(c, name)

    cls = name
    name = cls.__name__
    register_class(cls, name.lower())

    return cls

class Pipeline(Dataset):
    """
    A pipeline is used to read data to a magiCARP model during training. It handles data preprocessing and batching.
    """

    def __init__(self):
        super().__init__()

        # Preprocessing functions for the two different modalities
        # create_preprocess_fn must be called to set both
        self.prep_A : Callable[[Iterable[Any]], DataElement] = None
        self.prep_B : Callable[[Iterable[Any]], DataElement] = None
    
    @abstractclassmethod
    def __getitem__(self, idx : int) -> Tuple[Any, Any]:
        pass

    @abstractclassmethod
    def __len__(self) -> int:
        pass

    def create_preprocess_fns(self, call_feature_extractor : Callable[[Iterable[Any]],  Any], modality : str):
        """
        Function factory that generates and sets preprocessing functions for the pipeline. Default behavior is 
        to simply use call_feature_extractor as the entire preprocess function for given modality.

        :param call_feature_extractor: Function that takes in a batch of data and returns a batch of features. i.e. a tokenizer.
        :type call_feature_extractor: Callable[[Iterable[Any]],  Any]

        :param modality: Modality to create preprocessing function for. Must be either "A" or "B".
        :type modality: str
        """
    
        if modality == "A":
            self.prep_A = call_feature_extractor
        elif modality == "B":
            self.prep_B = call_feature_extractor
        else:
            raise ValueError("Modality must be either 'A' or 'B'.")

    def create_loader(self, device : torch.device = None, **kwargs) -> DataLoader:
        """
        Creates a dataloader for the pipeline.

        :param device: Device to load data to by default. If None, does nothing
        :type device: torch.device

        :param **kwargs: Any keyword arguments to pass to the DataLoader constructor.
        :type **kwargs: Any

        :return: A dataloader for the pipeline.
        :rtype: DataLoader
        """

        if self.prep_A is None or self.prep_B is None:
            raise ValueError("Preprocessing functions must be set before creating a dataloader.")
        
        def collate(batch : Iterable[Tuple[Any, Any]]) -> Tuple[DataElement, DataElement]:
            """
            Collates a batch of data into a tuple of tensors. This function is passed to the dataloader.
            """

            data_A, data_B = zip(*batch)
            data_A = list(data_A)
            data_B = list(data_B)

            data_A = self.prep_A(data_A)
            data_B = self.prep_B(data_B)

            if device is not None:
                data_A = data_A.to(device)
                data_B = data_B.to(device)
            
            return data_A, data_B

        return DataLoader(self, collate_fn = collate, **kwargs)