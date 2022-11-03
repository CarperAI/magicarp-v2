from abc import abstractclassmethod

from torch.optim import AdamW

from magicarp.configs import TrainConfig
from magicarp.models import CrossEncoder
from magicarp.pipeline  import Pipeline

class Trainer:
    def __init__(self, model : CrossEncoder, config : TrainConfig):
        self.model = model
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            eps=config.adam_epsilon
            )
        self.config = config

    def train(self, pipeline : Pipeline):
        epochs = self.config.num_epochs
    
        if pipeline.prep is None:
            pipeline.create_preprocess_fn(self.model.preprocess)
        
        loader = pipeline.create_loader(
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            shuffle=self.config.shuffle,
            pin_memory=self.config.pin_memory
        )

        for epoch in range(epochs):
            for batch in loader:
                y = self.model(batch)
                print(y)
                break


