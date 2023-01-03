from typing import Iterable
from torch import Tensor
import torch

from magicarp.pipeline.rpcd import InstructRPCD

from magicarp.models.imgtext import ImgTextEncoder
from magicarp.configs import magiCARPConfig
from magicarp.trainer.accelerate import AcceleratedTrainer

if __name__ == "__main__":
    pipe = InstructRPCD(path="data/rpcd", device = "cuda", min_comments=4, max_comments=9)

    # Load model
    config = magiCARPConfig.load_yaml("configs/imgtext_config.yml")
    model = ImgTextEncoder(config.model)

    trainer = AcceleratedTrainer(model, config.train)
    trainer.train(pipe)

