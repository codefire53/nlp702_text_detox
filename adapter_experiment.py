from omegaconf import DictConfig, OmegaConf
import hydra
import json
import numpy as np
from factory.adapter_model_factory import initialize_tokenizer
from utils import load_detox_sets, load_test_dataset, load_test_dataloader, load_detox_joint_sets
from datasets import load_dataset
from models.models import AdapterSeq2SeqModel, MoLoRAModel
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, RichProgressBar, LearningRateMonitor
from tqdm import tqdm
import csv

def merge_results(test_dataset, output_filepath, detox_filepath, toxic_filepath):
    output_path_prefix, output_path_suffix = output_filepath.rsplit('.', 1)
    detox_path_prefix, detox_path_suffix = detox_filepath.rsplit('.', 1)
    toxic_path_prefix, toxic_path_suffix = toxic_filepath.rsplit('.', 1)
    for lang_split in test_dataset:
        output_path = f"{output_path_prefix}_{lang_split}.{output_path_suffix}"
        detox_path = f"{detox_path_prefix}_{lang_split}.{detox_path_suffix}"
        toxic_path = f"{toxic_path_prefix}_{lang_split}.{toxic_path_suffix}"
        detox_sents = []
        with open(detox_path, 'r') as lines:
            for line in  lines:
                detox_sents.append(line.strip())
        
        toxic_sents = []
        with open(toxic_path, 'r') as lines:
            for line in  lines:
                toxic_sents.append(line.strip())

        test_pairs = [(toxic_sent, detox_sent) for toxic_sent, detox_sent in zip(toxic_sents, detox_sents)]
        with open(output_path, "w", newline='', encoding="utf-8") as file:
            writer = csv.writer(file, delimiter='\t')
            writer.writerows(test_pairs)

@hydra.main(version_base=None, config_path="./confs", config_name="adapter_experiment_config")
def main(cfg: DictConfig):
    seed_everything(42)
    if cfg.adapters.name == 'molora':
        model = MoLoRAModel(cfg.models, cfg.adapters, cfg.optim)
    else:
        model = AdapterSeq2SeqModel(cfg.models, cfg.adapters, cfg.optim)
    if cfg.dataset.use_joint:
        # initialize tokenizer for two languages
        model_config_params_lang1 = cfg.models.copy()
        model_config_params_lang1.multilang.target_id = cfg.dataset.lang1
        model_config_params_lang1.multilang.source_id = cfg.dataset.lang1
        tokenizer1 = initialize_tokenizer(model.model, model_config_params_lang1)
        model_config_params_lang2 = cfg.models.copy()
        model_config_params_lang2.multilang.target_id = cfg.dataset.lang2
        model_config_params_lang2.multilang.source_id = cfg.dataset.lang2
        tokenizer2 = initialize_tokenizer(model.model, model_config_params_lang2)

        train_dataloader, val_dataloader = load_detox_joint_sets(cfg.dataset, cfg.tokenizers, tokenizer1, tokenizer2, cfg.collators, cfg.dataloader)
    
    else:
        train_dataloader, val_dataloader = load_detox_sets(cfg.dataset, cfg.tokenizers, model.tokenizer, cfg.collators, cfg.dataloader)
    

    # wandb logger for monitoring
    wandb_logger = WandbLogger(**cfg.loggers)
    
    # callbacks
    checkpoint_callback = ModelCheckpoint(**cfg.checkpoint)
    early_stop = EarlyStopping(**cfg.earlystopping)
    rich = RichProgressBar()
    lr_monitor = LearningRateMonitor(**cfg.lr_monitors)
    callbacks = [checkpoint_callback, early_stop, lr_monitor, rich]

    trainer = Trainer(logger=wandb_logger, callbacks=callbacks, **cfg.trainer)

    if cfg.do_train:
        trainer.fit(model, train_dataloader, val_dataloader)
    
    if cfg.do_test:
        trainer = Trainer(logger=wandb_logger, accelerator="gpu", devices=1)

        src_sents = []
        langs = []
        with open(cfg.in_filepath, 'r', newline='') as f:
            reader = csv.reader(f, delimiter='\t')
            for idx, row in enumerate(reader):
                if idx == 0:
                    continue
                assert ((len(row) == 3) or (len(row) == 2))
                src_sents.append(row[0])
                langs.append(row[-1])
        if cfg.do_train:
            checkpoint_file = checkpoint_callback.best_model_path
        else:
            checkpoint_file = cfg.checkpoint_file

            
        if cfg.adapters.name == 'molora':
            model = MoLoRAModel.load_from_checkpoint(checkpoint_file, test_source=cfg.dataset.test_source, model_config_params=cfg.models, adapter_config_params=cfg.adapters, tokenizer=model.tokenizer, target_lang='en', pred_path=cfg.dataset.pred_filepath)
        else:
            model = AdapterSeq2SeqModel.load_from_checkpoint(checkpoint_file, test_source=cfg.dataset.test_source, model_config_params=cfg.models, adapter_config_params=cfg.adapters, tokenizer=model.tokenizer, target_lang='en', pred_path=cfg.dataset.pred_filepath)

            
        # This will output file for codalab submission
        if cfg.output_submission_file:
            with open(cfg.out_filepath, 'w', newline='') as file:
                writer = csv.writer(file, delimiter='\t')
                col_names = ['toxic_sentence', 'neutral_sentence', 'lang']
                writer.writerow(col_names)
                for src_sent, lang in zip(src_sents, langs):
                    tgt_sent = model.translate_sentence(src_sent, lang)
                    print(tgt_sent)
                    writer.writerow([src_sent, tgt_sent, lang])
        
        # This will output all jsons required for evaluation
        if cfg.output_self_eval_file:
            # use huggingface dataset for self evaluation
            test_dataset = load_test_dataset(cfg.dataset)
            prefix_src_eval_out, suffix_src_eval_out = cfg.out_src_eval_filepath.rsplit(".", 1)
            prefix_pred_eval_out, suffix_pred_eval_out = cfg.out_pred_eval_filepath.rsplit(".",1)
            if hasattr(cfg, 'target_field'):
                prefix_tgt_eval_out, suffix_tgt_eval_out = cfg.out_tgt_eval_filepath.rsplit(".", 1)
            for lang_split in test_dataset:
                lang_dataset = test_dataset[lang_split]
                id = 0
                src_dicts = []
                pred_dicts = []
                tgt_dicts = []
                if hasattr(cfg, 'target_field'):
                    for src_sent, tgt_sent in zip(lang_dataset[cfg.source_field], lang_dataset[cfg.target_field]):
                        pred_sent = model.translate_sentence(src_sent, lang_split)
                        src_dicts.append({"id": id, "text": src_sent})
                        pred_dicts.append({"id": id, "text": pred_sent})
                        tgt_dicts.append({"id": id, "text": tgt_sent})
                        idx += 1
                else:
                    for src_sent in lang_dataset[cfg.source_field]:
                        pred_sent = model.translate_sentence(src_sent, lang_split)
                        src_dicts.append({"id": id, "text": src_sent})
                        pred_dicts.append({"id": id, "text": pred_sent})
                        tgt_dicts.append({"id": id, "text": tgt_sent})
                        idx += 1

                src_eval_out = f"{prefix_src_eval_out}_{lang_split}.{suffix_src_eval_out}"
                with open(src_eval_out, 'w') as file:
                    for item in src_dicts:
                        json.dump(item, file)
                        file.write('\n')
                
                pred_eval_out = f"{prefix_pred_eval_out}_{lang_split}.{suffix_pred_eval_out}"
                with open(pred_eval_out, 'w') as file:
                    for item in pred_dicts:
                        json.dump(item, file)
                        file.write('\n')

                if hasattr(cfg, 'target_field'):
                    tgt_eval_out = f"{prefix_tgt_eval_out}_{lang_split}.{suffix_tgt_eval_out}"
                    with open(tgt_eval_out, 'w') as file:
                        for item in tgt_dicts:
                            json.dump(item, file)
                            file.write('\n')
            


if __name__ == "__main__":
    main()
