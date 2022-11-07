from typing import Iterable, Any
from torchtyping import TensorType
from transformers import AutoModel, AutoTokenizer

from magicarp.models import ModelOutput, CrossEncoder
from magicarp.configs import ModelConfig
from magicarp.data import TextElement

class TextTextEncoder(CrossEncoder):
    def __init__(self, config : ModelConfig):
        super().__init__(config)

        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)
        self.model = AutoModel.from_pretrained(config.model_path)

        # Add sep and cls tokens to the tokenizer
        self.tokenizer.add_tokens(["[SEP]", "[CLS]"])
        self.model.resize_token_embeddings(len(self.tokenizer))

    def preprocess(self, input_A : Iterable[str], input_B : Iterable[str]) -> Any:
        """
        Preprocess text into tokens without any padding or truncation.
        """
        
        # Concatenate each string in A to each string in B, separated by a sep token and add a cls token to end
        input_strings = [f"[CLS] {a} [SEP] {b}" for a, b in zip(input_A, input_B)]

        return self.tokenizer(
            input_strings,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length = self.config.sequence_length
            )

    def forward(self, x : TextElement):
        """ 
        Forward pass through the model.
        """
        print(x.input_ids.shape)
        out = self.model(
            input_ids=x.input_ids,
            attention_mask=x.attention_mask,
            output_hidden_states=True,
            return_dict=True
            )

        h : TensorType["batch", "seq_len", "d_model"] = out.hidden_states[-2]

        # Extract hidden state corresponding to CLS token (first token)
        cls_h : TensorType["batch", "d_model"] = h[:, 0, :]

        return ModelOutput(
           scores=self.score_head(h)
        )
    

