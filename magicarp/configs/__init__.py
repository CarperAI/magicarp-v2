from typing import Dict, Any, List, Optional

from dataclasses import dataclass
import yaml

@dataclass
class ModelConfig:
    """
    :param model_path: Path to the model checkpoint, local or on HF hub. If providing multiple paths, separate with a comma.
    :type model_path: str

    :param model_type: Name of registered model to use. This is used to determine the model class to use. If multiple, separate with a comma.
    :type model_type: str
    """
    
    model_path : str = None
    tokenizer_path : str = None
    model_type : Optional[str] = None
    sequence_length : int = 1024

    @classmethod
    def from_dict(cls, config: Dict[str, Any]):
        return cls(**config)

@dataclass
class TrainConfig:
    """

    """
    # Optimizer parameters
    learning_rate : float = 1e-4
    weight_decay : float = 0
    adam_epsilon : float = 1e-8

    # Scheduler parameters
    rampup_length : int = 400
    rampdown_length : int = 1000
    final_learning_rate : float = 1e-6

    # Training parameters
    num_epochs : int = 1
    batch_size : int = 16
    grad_accum_steps : int = 1
    max_grad_norm : float = None

    # Logging parameters
    log_interval : int = 100
    save_interval : int = 1000
    save_dir : str = None
    val_interval : int = 1000
    val_split : float = 0.1

    # Misc parameters
    device : str = "cuda"
    seed : int = 42
    num_workers : int = 0
    shuffle : bool = True
    pin_memory : bool = True

    # WANDB
    wandb_project : Optional[str] = None
    wandb_entity : Optional[str] = None

    @classmethod
    def from_dict(cls, config: Dict[str, Any]):
        return cls(**config)

@dataclass
class magiCARPConfig:
    model: ModelConfig
    train: TrainConfig

    @classmethod
    def load_yaml(cls, yml_fp: str):
        with open(yml_fp, mode="r") as file:
            config = yaml.safe_load(file)
        return cls(
            ModelConfig.from_dict(config["model"]),
            TrainConfig.from_dict(config["train"]),
        )

    def to_dict(self):
        data = self.model.__dict__.copy()
        data.update(self.train_job.__dict__)
        return data
