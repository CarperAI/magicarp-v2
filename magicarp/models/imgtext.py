
from typing import Iterable, Any, Tuple, Optional
from dataclasses import replace
from torchtyping import TensorType

from transformers import AutoModel, AutoTokenizer, AutoFeatureExtractor
from torch import nn
import torch.nn.functional as F
import torch

from axial_positional_embedding import AxialPositionalEmbedding

from PIL import Image

from magicarp.data import TextElement, ImageElement
from magicarp.configs import ModelConfig
from magicarp.models import CrossEncoder, ModelOutput

class ImgTextEncoder(CrossEncoder):
    """
    Image-text crossencoder using ViT.
    """

    def __init__(self, config : ModelConfig):
        # Assume config.model_path has path to ViT and LM, seperated by comma
        vit_path, model_path = config.model_path.split(",")

        # Want super to init with just the LM path
        self.config = config
        new_config = replace(config) # deep copy
        new_config.model_path = model_path
        super().__init__(config=new_config)

        # For ViT
        self.img_fe = AutoFeatureExtractor.from_pretrained(vit_path)
        self.img_embedder = AutoModel.from_pretrained(vit_path)

        for param in self.img_embedder.parameters():
            param.requires_grad = False

        # For LM
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)

        # Add sep and cls tokens to the tokenizer
        self.tokenizer.add_tokens(["[SEP]"])
        self.model.resize_token_embeddings(len(self.tokenizer))

        self.pad_token_id = self.tokenizer.pad_token_id

        # Project image embedding to text embedding size if needed
        if self.img_embedder.config.hidden_size != self.model.config.hidden_size:
            self.img_proj = nn.Linear(self.img_embedder.config.hidden_size, self.model.config.hidden_size)
        else:
            self.img_proj = None
        
        # How long should text be for it to be concatenated to image embeddings properly?
        self.n_patches : int = (self.img_embedder.config.image_size // self.img_embedder.config.patch_size) ** 2
        self.text_max_length : int = self.model.config.max_position_embeddings - self.n_patches - 1

        # 2d pos enc for ViT
        self.img_pos_enc = AxialPositionalEmbedding(
            dim = self.model.config.hidden_size,
            axial_shape = (int(self.n_patches ** 0.5), int(self.n_patches ** 0.5))
        )

    def preprocess(self, input_A : Iterable[Image.Image], input_B : Iterable[str]) -> Any:
        """
        Preprocess images into tensors and text into tokens without any padding or truncation.
        
        :param input_A: Input images
        :type input_A: Iterable[Image.Image]

        :param input_B: Input text
        :type input_B: Iterable[str]

        :return: Preprocessed inputs
        :rtype: Any
        """
        
        # Preprocess images
        img_inputs = self.img_fe(images=input_A, return_tensors="pt")

        # Preprocess text

        # Add sep token to start of each string in B
        # CLS token will be added to start later

        input_strings = [f"[SEP] {b}" for b in input_B]
        text_inputs = self.tokenizer(
            input_strings,
            padding='max_length',
            truncation=True,
            return_tensors="pt",
            max_length = self.model.config.max_position_embeddings
        )
    
        return img_inputs, text_inputs
    
    def forward(self, x : Tuple[ImageElement, TextElement], scores : Optional[TensorType["batch"]]):
        img : ImageElement = x[0]
        text : TextElement = x[1]

        # Extract image features
        img_features : TensorType["batch", "n_patches + 1", "d_model"] = self.img_embedder(**img.to_dict()).last_hidden_state
        if self.img_proj:
            img_features = self.img_proj(img_features)

        # 2d pos enc, skip cls token    
        img_features[1:] += self.img_pos_enc(img_features[1:])
        
        # Use embedding layer from LM alone
        text_features : TensorType["batch", "sequence_length", "d_model"] = self.model.embeddings(text.input_ids)
    
        text_features = text_features[:, :self.text_max_length, :]
        text.attention_mask = text.attention_mask[:, :self.text_max_length]
        # Stitch together
        embs : TensorType["batch", "sequence", "d_model"] = torch.cat([img_features, text_features], dim=1)

        # Create new attention mask and incorporate 0s from the text mask
        attn_mask : TensorType["batch", "sequence"] = torch.ones_like(embs[:, :, 0])
        attn_mask[:, self.n_patches + 1:] = text.attention_mask

        # Now through LM
        out = self.model(
            inputs_embeds=embs,
            attention_mask=attn_mask,
            output_hidden_states=True,
            return_dict=True
        )

        h : TensorType["batch", "seq_len", "d_model"] = out.hidden_states[-2]

        # Extract hidden state corresponding to CLS token (first token)
        cls_h : TensorType["batch", "d_model"] = h[:, 0, :]

        scores_pred = self.score_head(cls_h).squeeze()
        loss = None
        if scores is not None:
            loss = F.mse_loss(scores_pred, scores)
            
        return ModelOutput(
           scores=scores_pred,
           loss=loss
        )        
