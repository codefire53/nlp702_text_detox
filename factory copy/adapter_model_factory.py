from omegaconf import DictConfig
from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel
from peft import get_peft_model


def initialize_adapter_model(model_config_params: DictConfig, adapter_type: str, adapter_config):
    model_config_params_copy = model_config_params.copy()
    if hasattr(model_config_params_copy, "multilang"):
        del model_config_params_copy.multilang
    model = AutoModelForSeq2SeqLM.from_pretrained(**model_config_params_copy)
    model = get_peft_model(model, adapter_config)
    return model


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