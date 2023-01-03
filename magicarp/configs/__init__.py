from typing import Dict, Any, List, Optional, Union

from dataclasses import dataclass
import yaml

@dataclass
class ModelConfig:
    """
    :param model_path: Path to the model checkpoint, local or on HF hub. If providing multiple paths, separate with a comma.
    :type model_path: str

    :param model_type: Name of registered model to use. This is used to determine the model class to use. If multiple, separate with a comma.
    :type model_type: str

    :param sequence_length: Maximum sequence length for the model. May not always be used over model_position_embeddings
    :type sequence_length: int

    :param embed_method: Method to use for embedding. Options are cls, mean, masked_sum
    :type embed_method: str

    :param unfrozen_layers: Freeze all layers in the LM except last N. If float is provided, it is interpreted as a proportion of layers to unfreeze.
    :type unfrozen_layers: Union[int, float]
    """
    
    model_path : str = None
    tokenizer_path : str = None
    model_type : Optional[str] = None
    sequence_length : int = 1024
    embed_method : str = None
    unfrozen_layers : Union[int, float] = None

    @classmethod
    def from_dict(cls, config: Dict[str, Any]):
        return cls(**config)

@dataclass
class TrainConfig:
    """
    Configuration for trainer.

    :param loss_type: Type of loss to use. Options are ranking, pairwise, densepairwise
    :type loss_type: str

    :param reward_type: Type of reward to use. Options are End or Step. End means reward is computed at final step, Step means reward is computed at every step.
    :type reward_type: str

    :param query_modality: Which modality to use as query (as opposed to response). Options are A or B. I.e. in image generation the prompt is the query.
    :type query_modality: str

    :param val_batch_multiplier: Multiplier for batch size during validation.
    """
    # Loss
    loss_type : str = None
    reward_type : str = "End"
    query_modality : str = "A"

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
    val_batch_multiplier : int = 1

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
