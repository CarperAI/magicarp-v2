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

        # Preprocessing function for both modalities
        # Can return a single data element or multiple
        self.prep : Callable[[Iterable[Any], Iterable[Any]], Iterable[DataElement]]
    
    @abstractclassmethod
    def __getitem__(self, idx : int) -> Tuple[Any, Any]:
        pass

    @abstractclassmethod
    def __len__(self) -> int:
        pass

    def create_preprocess_fn(self, call_feature_extractor : Callable[[Iterable[Any], Iterable[Any]],  Any]):
        """
        Function factory that generates and sets preprocessing function for the pipeline. Default behavior is 
        to simply use call_feature_extractor as the entire preprocess function.

        :param call_feature_extractor: Function that takes in a batch of data and returns a batch of features. i.e. a tokenizer.
        :type call_feature_extractor: Callable[[Iterable[Any]],  Any]

        """
    
        self.prep = call_feature_extractor

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

        if self.prep is None:
            raise ValueError("Preprocessing function must be set before creating a dataloader.")
        
        def collate(batch : Iterable[Tuple[Any, Any]]) -> Tuple[DataElement, DataElement]:
            """
            Collates a batch of data into a tuple of tensors. This function is passed to the dataloader.
            """

            data_A, data_B = zip(*batch)
            data_A = list(data_A)
            data_B = list(data_B)

            res = self.prep(data_A, data_B)
            if type(res) is list or type(res) is tuple:
                res = [r.to(device) for r in res]
            else:
                res = res.to(device)
            
            return res

        return DataLoader(self, collate_fn = collate, **kwargs)

    @abstractclassmethod
    def partition_validation_set(self, val_size : float = 0.1, shuffle : bool = False):
        """
        Partitions the dataset into a training and validation set. This function should be called before creating a dataloader.

        :param val_size: The proportion of the dataset to use for validation. Defaults to 0.1.
        :type val_size: float

        :param shuffle: Whether to shuffle the dataset before partitioning. Defaults to True.
        :type shuffle: bool
        """
        pass

    @abstractclassmethod
    def create_validation_loader(self, device : torch.device = None, **kwargs) -> DataLoader:
        """
        Create a dataloader for validation data. 
        """
        pass
