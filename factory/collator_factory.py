from transformers import AutoTokenizer, DataCollatorForSeq2Seq, DataCollator
from omegaconf import DictConfig
from adapters import AutoAdapterModel

def initialize_collator_module(tokenizer: AutoTokenizer, model: AutoAdapterModel, max_length: int, collator_config_params: DictConfig) -> DataCollator:
    return DataCollatorForSeq2Seq(tokenizer, model=model, max_length=max_length, **collator_config_params)