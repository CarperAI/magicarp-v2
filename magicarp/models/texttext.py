from typing import Iterable, Any, Tuple, Optional
from dataclasses import replace
from torchtyping import TensorType

from transformers import AutoModel, AutoTokenizer, AutoFeatureExtractor
from torch import nn
import torch.nn.functional as F
import torch
import einops as eo

from magicarp.data import TextElement
from magicarp.configs import ModelConfig
from magicarp.models import CrossEncoder, ModelOutput

class TextTextEncoder(CrossEncoder):
    """
    Text-text cross encoder
    """

    def __init__(self, config : ModelConfig):
        model_path = config.model_path

        self.config = config
        # inits model and tokenizer
        super().__init__(config=config)

        self.pad_token_id = self.tokenizer.pad_token_id
        self.text_max_length : int = self.model.config.max_position_embeddings

    def preprocess(self, input_A : Iterable[str], input_B : Iterable[str]) -> Any:
        """
        Preprocess two strings into joined token sequence
        """

        input_A = [f"[CLS] {text}" for text in input_A]
        input_B = [f"[SEP] {text}" for text in input_B]

        input_A = self.tokenizer(
            input_A,
            padding=True,
            truncation=True,
            max_length=self.text_max_length // 2,
            return_tensors="pt"
        )

        is_even = self.text_max_length % 2 == 0

        # Leaving room for CLS token is different if max size is even
        input_B = self.tokenizer(
            input_B,
            padding=True,
            truncation=True,
            max_length=self.text_max_length // 2 + 1 if is_even else self.text_max_length // 2,
            return_tensors="pt"
        )

        return input_A, input_B

    def forward(
        self,
        x : Tuple[TextElement, TextElement],
        scores : Optional[TensorType["batch"]] = None
    ) -> ModelOutput:

        text_A = x[0]
        text_B = x[1]

        attn_mask : TensorType["b", "n"] = torch.cat([text_A.attention_mask, text_B.attention_mask], dim=1)
        features : TensorType["b", "n", "d"] = self.model(
            input_ids=torch.cat([text_A.input_ids, text_B.input_ids], dim=1),
            attention_mask=attn_mask,
            output_hidden_states=True,
            return_dict=True
        )

        h : TensorType["b", "n", "d"] = features.hidden_states[-2]
        h = self.embed(h, attn_mask)

        scores_pred = self.score_head(h).squeeze()
        loss = None
        if scores is not None:
            loss = self.loss_fn(scores_pred, scores)
        
        return ModelOutput(
            scores=scores_pred,
            loss=loss
        )


        