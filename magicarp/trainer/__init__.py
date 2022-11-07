from abc import abstractclassmethod

from torch import optim
import torch
import wandb
import os

from magicarp.configs import TrainConfig
from magicarp.models import CrossEncoder, ModelOutput
from magicarp.pipeline  import Pipeline
from magicarp.utils import get_intervals, wandb_start

class Trainer:
    def __init__(self, model : CrossEncoder, config : TrainConfig):
        self.model : torch.nn.Module = model
        self.config : TrainConfig = config

        self.model.to(self.config.device)

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            eps=config.adam_epsilon
            )

        rampup_Length = config.rampup_length
        rampdown_Length = config.rampdown_length
        final_learning_rate = config.final_learning_rate

        # Scheduler with warmup and decay
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.learning_rate,
            total_steps=rampup_Length + rampdown_Length,
            pct_start=rampup_Length / (rampup_Length + rampdown_Length),
            final_div_factor=final_learning_rate / config.learning_rate,
            anneal_strategy="linear"
        )
    
    def load_checkpoint(self, path : str):
        try:
            self.model.load_state_dict(torch.load(f"{path}/model.pt"))
            self.optimizer.load_state_dict(torch.load(f"{path}/optimizer.pt"))
            self.scheduler.load_state_dict(torch.load(f"{path}/scheduler.pt"))
        except:
            print(f"Trainer could not load checkpoint from {path}")
    
    def save_checkpoint(self, path : str):
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        
        torch.save(self.model.state_dict(), f"{path}/model.pt")
        torch.save(self.optimizer.state_dict(), f"{path}/optimizer.pt")
        torch.save(self.scheduler.state_dict(), f"{path}/scheduler.pt")

    def train(self, pipeline : Pipeline):
        epochs = self.config.num_epochs
        use_wandb = False
        do_val = False
        do_save = False if self.config.save_dir is None else True

        if self.config.wandb_project is not None:
            wandb_start(self.config)
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
            for i, (a, b, scores) in enumerate(loader):
                self.optimizer.zero_grad()

                y : ModelOutput = self.model((a, b), scores)
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
            for i, (a, b, scores) in enumerate(loader):
                y : ModelOutput = self.model((a, b), scores)
                loss = y.loss
                avg_loss += loss

        avg_loss /= steps
        print(f"Validation Loss: {avg_loss}")
        if self.config.wandb_project is not None:
            wandb.log({"val_loss": avg_loss})


        


                








