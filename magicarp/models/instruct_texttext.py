from typing import Iterable, Any
from torchtyping import TensorType
from transformers import AutoModel, AutoTokenizer
import torch

from magicarp.models import ModelOutput
from magicarp.models.texttext import TextTextEncoder
from magicarp.configs import ModelConfig
from magicarp.data import DataElement

class InstructTextText(TextTextEncoder):
    """
    Implements a text-text cross encoder with ranking based loss ala InstructGPT
    """
    def forward(self, x : Iterable[DataElement], compute_loss : bool = False) -> ModelOutput:
        """ 
        :param x: List consisting of two TextElements. Either the first must be of length one, or both must be of equal length. 
        """
        k = len(x[0].input_ids) # length of rankings, i.e. top k
        pairwise : bool = len(x[1].input_ids) == k

        if not pairwise and len(x[0].input_ids) != 1:
            raise ValueError("If not pairwise, the first element must be of length one")

        if not pairwise:
            x[0].input_ids = x[0].input_ids.repeat(k, 1)
            x[0].attention_mask = x[0].attention_mask.repeat(k, 1)
        
        scores : TensorType["k"] = super().forward(x, scores = None).scores
        out = ModelOutput(scores = scores)

        if not compute_loss:
            return out
        
        # Check instruct_imgtext.py for more info on below computations
        n_comparisons = k * (k - 1) // 2
        S = scores.unsqueeze(1) - scores.unsqueeze(0)
        S = torch.triu(S)
        S = torch.where(S != 0, S.sigmoid().log(), torch.zeros_like(S))
        out.loss = -1 * S.sum() / n_comparisons

        return out

    

