import torch
import wandb

from magicarp.models import ModelOutput
from magicarp.pipeline import Pipeline
from magicarp.trainer import Trainer
from magicarp.utils import wandb_start, get_intervals

class InstructTrainer(Trainer):
    """
    Trainer that trains reranker directly from rankings without scores
    """

    def train(self, pipeline : Pipeline):
        epochs = self.config.num_epochs
        use_wandb = False
        do_val = False
        do_save = False if self.config.save_dir is None else True

        if self.config.wandb_project is not None:
            wandb_start(self.config)
            wandb.watch(self.model)
            use_wandb = True

        if pipeline.prep is None:
            pipeline.create_preprocess_fn(self.model.preprocess)
        
        if self.config.val_split > 0:
            do_val = True
            pipeline.partition_validation_set(self.config.val_split)
        
        loader = pipeline.create_loader(
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            shuffle=self.config.shuffle,
            pin_memory=self.config.pin_memory
        )

        for epoch in range(epochs):
            for i, data in enumerate(loader):
                self.optimizer.zero_grad()

                y : ModelOutput = self.model(data, compute_loss = True)
                loss = y.loss
                loss.backward()

                self.optimizer.step()
                self.scheduler.step()

                intervals = get_intervals(self.config, i)

                if intervals["log"]:
                    print(f"Epoch {epoch} | Batch {i} | Loss {loss}")
                    if use_wandb:
                        wandb.log({"loss": loss})
                
                if intervals["save"] and do_save:
                    self.save_checkpoint(self.config.save_dir)
                
                if intervals["val"] and do_val:
                    self.validate(pipeline)

    def validate(self, pipeline : Pipeline):
        loader = pipeline.create_validation_loader(
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            shuffle=False,
            pin_memory=self.config.pin_memory
        )

        avg_loss = 0
        steps = len(loader)

        with torch.no_grad():
            for i, data in enumerate(loader):
                y : ModelOutput = self.model(data, compute_loss = True)
                loss = y.loss
                avg_loss += loss

        avg_loss /= steps
        print(f"Validation Loss: {avg_loss}")
        if self.config.wandb_project is not None:
            wandb.log({"val_loss": avg_loss})