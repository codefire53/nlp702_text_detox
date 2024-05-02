from omegaconf import DictConfig
from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel
from peft import get_peft_model
from adapters import AutoAdapterModel
from adapters.composition import Fuse

def add_adapter_fusion(model, adapter_config):
    for adapter_name in adapter_config.adapter_names:
        model.add_adapter(adapter_name, config="seq_bn")
    adapter_setup = Fuse(*adapter_config.adapter_names)
    model.add_adapter_fusion(adapter_setup)
    model.set_active_adapters(adapter_setup)
    model.add_seq2seq_head("paradetox")
    model.train_adapter_fusion(adapter_setup)


def initialize_adapter_model(model, adapter_config):
    model = get_peft_model(model, adapter_config)
    return model

def initialize_model(model_config_params: DictConfig):
    if hasattr(model_config_params, "is_adapter_model") and model_config_params.is_adapter_model:
        config = AutoConfig.from_pretrained(model_config_params.pretrained_model_name_or_path)
        model = AutoAdapterModel.from_pretrained(model_config_params.pretrained_model_name_or_path, config=config)
    else:
        return AutoModelForSeq2SeqLM.from_pretrained(model_config_params.pretrained_model_name_or_path)

def initialize_tokenizer(model: AutoModel, model_config_params: DictConfig) -> AutoTokenizer:
    if hasattr(model_config_params, "multilang"):
        model_config_params_copy = model_config_params.copy()
        del model_config_params_copy.multilang
        tokenizer = AutoTokenizer.from_pretrained(**model_config_params_copy)
        if model.config.decoder_start_token_id is None:
            model.config.decoder_start_token_id = tokenizer.lang_code_to_id[model_config_params.multilang.target_id]
        tokenizer.src_lang = model_config_params.multilang.source_id
        tokenizer.set_src_lang_special_tokens(tokenizer.src_lang)
        tokenizer.tgt_lang = model_config_params.multilang.target_id
       
    else:
        tokenizer = AutoTokenizer.from_pretrained(**model_config_params)
    return tokenizer
