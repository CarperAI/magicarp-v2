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
            Input should be ImageElement followed by TextElement.
            Varies behavior depending on number of images provided.
            If number of images matches number of text elements, then
            each image is paired with its corresponding text element and score is computed.
            If number of images is 1, then the image is paired with all text elements.
            If compute loss is true, the second case is assumed, and loss for all comparisons
            is computed.
        """
        x_img : ImageElement = x[0]
        x_txt : TextElement = x[1]

        k = len(x_txt.input_ids) # length of rankings, i.e. top k
        pairwise : bool = len(x_img.pixel_values) == k

        if not pairwise:
            x_img.pixel_values = x_img.pixel_values.repeat(k, 1, 1, 1)
        
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



        
        

