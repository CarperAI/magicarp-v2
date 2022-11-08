from typing import Tuple, Callable, Iterable, Any
from torchtyping import TensorType

from datasets import load_from_disk
import os
import joblib
from torch import Tensor

from magicarp.pipeline import Pipeline, register_datapipeline
from magicarp.data import TextElement

@register_datapipeline
class StoryCritiquePipeline(Pipeline):
    """
    Pipeline for the Story-Critique dataset in the original CARP paper.

    :param path: Path to the Story-Critique dataset in arrow format.
    """
    def __init__(self, path : str):
        super().__init__()

        self.prep : Callable[[Iterable[str], Iterable[str]], TextElement] = None

        dataset = load_from_disk(path)
        train = dataset["train"]
        self.passages = train["story_target"]
        self.reviews = train["target_comment"]

        # As an additional preprocessing step, we filter out passages/reviews that are too short (<= 7 characters)
        # To save time on consecutive runs, we save indices of passages/reviews that passed filter
        # If the file doesn't exist, we create it and run filter
        if os.path.exists("data/story_critique/inds.joblib"):
            inds = joblib.load("data/story_critique/inds.joblib")
        else:
            inds = []
            for i in range(len(self.passages)):
                if len(self.passages[i]) > 7 and len(self.reviews[i]) > 7:
                    inds.append(i)
            joblib.dump(inds, "data/story_critique/inds.joblib")
        
        self.passages = [self.passages[i] for i in inds]
        self.reviews = [self.reviews[i] for i in inds]

        if len(self.passages) != len(self.reviews):
            raise ValueError("Passages and reviews must be the same length.")

    def __getitem__(self, index: int) -> Tuple[str, str]:
        return self.passages[index], self.reviews[index]

    def __len__(self) -> int:
        return len(self.passages)
    
    def create_preprocess_fn(self, call_feature_extractor: Callable[[Iterable[Any]], Any]):
        def prep(batch_A : Iterable[str], batch_B : Iterable[str]) -> TextElement:
            return TextElement(**call_feature_extractor(batch_A, batch_B))

        self.prep = prep
        



