from dataclasses import dataclass

@dataclass
class ModelConfig:
    """
    :param model_path: Path to the model checkpoint, local or on HF hub. If providing multiple paths, separate with a comma.
    :type model_path: str

    :param model_type: Name of registered model to use. This is used to determine the model class to use. If multiple, separate with a comma.
    :type model_type: str
    """
    
    model_path : str = None
    model_type : str = None

@dataclass
class TrainConfig:
    """
    :param pipeline: Name of pipeline to use.
    :type pipeline: str
    """

    pipeline : str = None