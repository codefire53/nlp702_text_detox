from omegaconf import DictConfig, OmegaConf
import hydra
from transformers import DataCollatorForSeq2Seq
from adapters import Seq2SeqAdapterTrainer, AutoAdapterModel
from factory.adapter_config_factory import *
from factory.adapter_model_factory import *
from factory.callback_factory import *
from factory.collator_factory import *
from factory.dataset_factory import *
from transformers import Seq2SeqTrainingArguments
import numpy as np



@hydra.main(version_base=None, config_path="./confs", config_name="adapter_experiment_config")
def train(cfg: DictConfig):
    tokenizer = initialize_tokenizer(cfg.models)
    print(OmegaConf.to_yaml(cfg))
    adapter_config = create_adapter_config(cfg.adapters)


    model = initialize_model(cfg.models, cfg.adapters, adapter_config)
    training_args = Seq2SeqTrainingArguments(run_name=cfg.run_name, output_dir=cfg.output_dir, logging_dir=cfg.logging_dir, **cfg.train)
    
    train_dataset, val_dataset = load_detox_train_sets(cfg.dataset, cfg.tokenizers, tokenizer)
    total_trainable_params = 0
    for _, p in model.named_parameters():
        total_trainable_params += p.requires_grad==True
    collator = initialize_collator_module(tokenizer, model, cfg.tokenizers.max_length, cfg.collators)
    print(f"Total trainable params: {total_trainable_params} params")
    trainer = Seq2SeqAdapterTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collator,
    tokenizer=tokenizer,
    callbacks=[initialize_early_stopping_module(cfg.earlystopping)]
    )
    trainer.train()

if __name__ == "__main__":
    train()
