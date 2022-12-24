from abc import abstractmethod
from typing import Optional, Any, Dict
from torchtyping import TensorType

import sys
import torch

from magicarp.data import DataElement
from magicarp.models import ModelOutput
from magicarp.losses import register_loss, EndLoss, StepLoss

@register_loss
class Pairwise(EndLoss):
    """
        Assuming first modality is A and second modality is B, this loss is intended for data where humans are given one A and two choices for B. Scores should include
        model output for A with accepted B and A with rejected B. The scores provided should be an even length tensor with the first half containing the scores for the
        accepted Bs and the second half containing the scores for the rejected Bs.
    """

    def __forward__(
        self,
        model_out : ModelOutput,
        expected_scores : Optional[TensorType["batch"]] = None
    ) -> ModelOutput:

        bs = len(model_out.scores) // 2
        scores = model_out.scores

        chosen_scores = scores[:bs]
        rejected_scores = scores[bs:]
        diff = chosen_scores - rejected_scores

        model_out.loss = -1 * diff.sigmoid().log().sum() / bs
        return model_out
    
@register_loss
class DensePairwise(StepLoss):
    """
        Refer to Pairwise for details on kinds of data this loss is meant for. As opposed to normal pairwise, incorporates input sequence
        into loss computatation, computing reward for each token in the sequence. Note that the input sequence is 
    """

    def weighted_sum(self, scores : TensorType["b", "n"]) -> TensorType["b"]:
        """
            How to weigh losses for computed scores with respect to seqeuence position.
            Defaults to weighing each uniformally (i.e. 1/n)
        """
        return scores.mean(1)

    def forward(
        self,
        model_out : ModelOutput,
        input_sequence : Optional[DataElement] = None,
        expected_scores : Optional[TensorType["batch"]] = None
    ) -> ModelOutput:
        bs = len(model_out.scores) // 2
        scores : TensorType["bs * 2, n"] = model_out.scores

        n = scores.shape[1]

        # Assuming model processes sequence of an even length,
        # first half is A, second half is [SEP] B. We only care about rewards for the query

        chosen_scores : TensorType["b, n"]= scores[:bs]
        rejected_scores : TensorType["b, n"] = scores[bs:]

        if self.query == "A":
            chosen_scores = chosen_scores[:, :n//2]
            rejected_scores = rejected_scores[:, :n//2]
        elif self.query == "B":
            # Skip sep token
            chosen_scores = chosen_scores[:, n//2 + 1:]
            rejected_scores = rejected_scores[:, n//2 + 1:]

        diff = chosen_scores - rejected_scores
        model_out.loss = self.weighted_sum(-1 * diff.sigmoid().log())

        return model_out
