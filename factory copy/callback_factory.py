from transformers import EarlyStoppingCallback, TrainerCallback
from omegaconf import DictConfig

def initialize_early_stopping_module(callback_config_params: DictConfig) -> TrainerCallback:
    return EarlyStoppingCallback(**callback_config_params)