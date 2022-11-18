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

    # Loss function from passage re-ranking with BERT
    # In this case, relevant passages (comments) are those with many upvotes, irrelevant are those
    # with many downvotes. Dataset is biased towards passages with many upvotes or downvotes
    # So we assume  from this that passages are strongly relevant or irrelevant
    def loss_fn(pred_scores, true_scores):

        # mse
        loss = torch.mean((pred_scores - true_scores)**2)
        return loss

        #relevant = true_scores > 0.5
        #pred_scores = torch.sigmoid(pred_scores) # Sigmoid to get probabilities
        #loss = torch.where(relevant, torch.log(pred_scores), torch.log(1 - pred_scores))

        #return -1 * loss.sum()

    config = magiCARPConfig.load_yaml("configs/imgtext_config.yml")

    model = InstructImgText(config.model)

    trainer = InstructTrainer(model, config.train)
    trainer.train(pipe)

