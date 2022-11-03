from typing import Iterable
from torch import Tensor
import torch

from magicarp.data import TextElement
from magicarp.pipeline.story_critique import StoryCritiquePipeline

pipe = StoryCritiquePipeline("data/story_critique")

# Load model

from magicarp.models.texttext import TextTextEncoder
from magicarp.configs import magiCARPConfig
from magicarp.trainer import Trainer

config = magiCARPConfig.load_yaml("configs/base_config.yml")
model = TextTextEncoder(config.model)
trainer = Trainer(model, config.train)
trainer.train(pipe)
