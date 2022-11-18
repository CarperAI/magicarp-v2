from typing import Iterable
from torch import Tensor
import torch

from magicarp.pipeline.rpcd import InstructRPCD

from magicarp.models.instruct_imgtext import InstructImgText
from magicarp.configs import magiCARPConfig
from magicarp.trainer.instruct_trainer import InstructTrainer

if __name__ == "__main__":
    pipe = InstructRPCD(path="data/rpcd", device = "cuda", min_comments=4, max_comments=9)

    # Load model
    config = magiCARPConfig.load_yaml("configs/imgtext_config.yml")
    model = InstructImgText(config.model)

    trainer = InstructTrainer(model, config.train)
    trainer.train(pipe)

