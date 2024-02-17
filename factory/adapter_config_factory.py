
from adapters import AdapterConfig, SeqBnConfig, UniPELTConfig
from omegaconf import DictConfig

def create_adapter_config(adapter_config_params: DictConfig) -> AdapterConfig:
    adapter_type = adapter_config_params.name
    config_params_copy = adapter_config_params.copy()
    del config_params_copy.name
    if 'pfeiffer' in adapter_type:
        return SeqBnConfig(**config_params_copy)
    else:
        return UniPELTConfig()