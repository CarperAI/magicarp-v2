from typing import Iterable
from torch import Tensor
import torch

from magicarp.data import TextElement
from magicarp.pipeline.rpcd import RPCDPipeline

pipe = RPCDPipeline("data/rpcd", device = "cuda", min_comments=10)

# Load model

from magicarp.models.imgtext import ImgTextEncoder
from magicarp.configs import magiCARPConfig
from magicarp.trainer import Trainer

config = magiCARPConfig.load_yaml("configs/imgtext_config.yml")
model = ImgTextEncoder(config.model)
trainer = Trainer(model, config.train)
trainer.train(pipe)
