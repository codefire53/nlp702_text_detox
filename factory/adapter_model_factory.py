from omegaconf import DictConfig
from adapters import AdapterConfig, AutoAdapterModel
from transformers import AutoConfig, AutoTokenizer
from adapters.composition import Stack

def initialize_model(model_config_params: DictConfig, adapter_config_params: DictConfig, adapter_config: AdapterConfig):

    config = AutoConfig.from_pretrained(
        **model_config_params
    )
    model = AutoAdapterModel.from_pretrained(
        config=config,
        **model_config_params
    )

    model.add_adapter(adapter_config_params.name, config=adapter_config)
    model.add_seq2seq_lm_head(adapter_config_params.name)
    model.train_adapter([adapter_config_params.name])
    return model

def initialize_tokenizer(model_config_params: DictConfig) -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(**model_config_params)
