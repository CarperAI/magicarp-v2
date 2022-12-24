from abc import abstractmethod
from typing import Optional, Any, Dict
from torchtyping import TensorType

import sys
import torch

from magicarp.models import ModelOutput
from magicarp.losses import register_loss, EndLoss

@register_loss
class InstructRanking(EndLoss):
    """
        InstructGPT type ranking loss. Assuming A is the data for the first modality and B is the data for the second modality, this loss is intended for cases where
        batches consist of a single A and an ordering of Bs based on human preference based ranking of the Bs for how well they "fit" A. Inputs to the loss are ordered
        (w.r.t the human ranking) scores that compare the single A against each B.
    """

    def forward(
        self,
        model_out : ModelOutput,
        expected_scores : Optional[TensorType["batch"]] = None
    ) -> ModelOutput:
        """
            Compute loss given scores. Some losses may require expected scores, while others are entirely unsupervised.

            :param scores: Scores predicted by model
            :type scores: TensorType["batch"]

            :param expected_scores: Expected scores, defaults to None
            :type expected_scores: Optional[TensorType["batch"]]

            :return: ModelOutput with loss and scores from model
            :rtype: ModelOutput
        """
        scores = model_out.scores
        k = len(scores)
        n_comparisons = k * (k - 1) // 2

        # Create a matrix S, where S[i, j] = scores[i] - scores[j]
        S = scores.unsqueeze(1) - scores.unsqueeze(0)
        # Only interested in S[i,j] where i < j, so everything above the upper diagonal
        S = torch.triu(S)
        
        S = torch.where(S != 0, S.sigmoid().log(), torch.zeros_like(S))
        model_out.loss = -1 * S.sum() / n_comparisons

        return model_out