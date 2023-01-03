from accelerate import Accelerator
import wandb
import torch

from magicarp.models import CrossEncoder, ModelOutput
from magicarp.pipeline import Pipeline
from magicarp.configs import TrainConfig
from magicarp.trainer import Trainer
from magicarp.utils import get_intervals, wandb_start

class AcceleratedTrainer(Trainer):
    def __init__(self, model : CrossEncoder, config : TrainConfig):
        super().__init__(model, config)
        self.accelerator = Accelerator()

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

        accelerator = Accelerator()
        self.model, self.optimizer, loader, self.scheduler = accelerator.prepare(
            self.model, self.optimizer, loader, self.scheduler
        )

        for epoch in range(epochs):
            for i, (a, b) in enumerate(loader):
                self.optimizer.zero_grad()

                y : ModelOutput = self.loss(self.model((a, b)))

                loss = y.loss
                accelerator.backward(loss)

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
        if pipeline.val_set is not None:
            pipeline.val_set.prep = pipeline.prep
            
        loader = pipeline.create_validation_loader(
            batch_size=self.config.batch_size * self.config.val_batch_multiplier,
            num_workers=self.config.num_workers,
            shuffle=False,
            pin_memory=self.config.pin_memory
        )

        loader = self.accelerator.prepare(loader)

        avg_loss = 0
        steps = len(loader)

        self.model.eval()
        with torch.no_grad():
            for i, (a, b) in enumerate(loader):
                y : ModelOutput = self.loss(self.model((a, b)))
                loss = y.loss
                avg_loss += loss
        self.model.train()

        avg_loss /= steps
        print(f"Validation Loss: {avg_loss}")
        if self.config.wandb_project is not None:
            wandb.log({"val_loss": avg_loss})