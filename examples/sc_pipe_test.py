from transformers import AutoTokenizer
from typing import Iterable
from torch import Tensor
import torch

from magicarp.data import TextElement
from magicarp.pipeline.story_critique import StoryCritiquePipeline

pipe = StoryCritiquePipeline("data/story_critique")

tok = AutoTokenizer.from_pretrained("roberta-large")
tok.add_tokens(["[quote]"])

def call_tok(batch : Iterable[str]) -> Tensor:
    return tok(batch, padding=True, truncation=True, max_length = 512, return_tensors="pt")

pipe.create_preprocess_fns(call_tok, "A")
pipe.create_preprocess_fns(call_tok, "B")

# Create loader
loader = pipe.create_loader(device = 'cuda', batch_size=16, shuffle=True, num_workers=0)

for batch in loader:
    pass_, rev_ = batch

    assert type(pass_) == TextElement
    assert type(rev_) == TextElement

    assert type(pass_.input_ids) == Tensor
    assert type(pass_.attention_mask) == Tensor

    assert type(rev_.input_ids) == Tensor
    assert type(rev_.attention_mask) == Tensor
    break