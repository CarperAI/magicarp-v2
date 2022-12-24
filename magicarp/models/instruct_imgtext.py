from typing import Iterable

from torchtyping import TensorType
import einops as eo
import torch

from magicarp.data import DataElement, ImageElement, TextElement
from magicarp.models import ModelOutput
from magicarp.models.imgtext import ImgTextEncoder

class InstructImgText(ImgTextEncoder):
    def forward(self, x : Iterable[DataElement], compute_loss : bool = False) -> ModelOutput:
        """
            Forward pass through the model, computes scores for each image-text pair. If compute_loss is True, also computes loss,
            given some loss function.
        """
        x_img : ImageElement = x[0]
        x_txt : TextElement = x[1]

        if len(x_img) != len(x_txt):
            raise ValueError("Number of images and text elements must match")
        
        scores : TensorType["k"] = super().forward((x_img, x_txt), scores = None).scores 
        out = ModelOutput(scores = scores)

        if not compute_loss:
            return out
        
        # Compute loss by creating a tensor of all possible scores comparisons
        # The score earlier in the batch always comes first
        n_comparisons = k * (k - 1) // 2

        # Create a matrix S, where S[i, j] = scores[i] - scores[j]
        S = scores.unsqueeze(1) - scores.unsqueeze(0)
        # Only interested in S[i,j] where i < j, so everything above the upper diagonal
        S = torch.triu(S)
        
        S = torch.where(S != 0, S.sigmoid().log(), torch.zeros_like(S))
        out.loss = -1 * S.sum() / n_comparisons

        return out



        
        

