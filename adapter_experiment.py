from omegaconf import DictConfig, OmegaConf
import hydra
import numpy as np
from factory.adapter_model_factory import initialize_tokenizer
from utils import load_detox_sets, load_test_dataset, load_test_dataloader
from datasets import load_dataset
from models.models import AdapterSeq2SeqModel
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

    model = AdapterSeq2SeqModel(cfg.models, cfg.adapters, cfg.optim)
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
        test_dataset = load_test_dataset(cfg.dataset)
        for lang_split in test_dataset:
            curr_test_dataset = test_dataset[lang_split]
            test_dataloader = load_test_dataloader(curr_test_dataset, cfg.dataset, cfg.tokenizers, model.tokenizer, cfg.collators, cfg.dataloader, lang_split)
            if cfg.do_train:
                # load best checkpoint
                model = AdapterSeq2SeqModel.load_from_checkpoint(checkpoint_callback.best_model_path, test_source=cfg.dataset.test_source, model_config_params=cfg.models, adapter_config_params=cfg.adapters, tokenizer=model.tokenizer, target_lang=lang_split, pred_path=cfg.dataset.pred_filepath)
            elif cfg.checkpoint_file:
                model = AdapterSeq2SeqModel.load_from_checkpoint(cfg.checkpoint_file, test_source=cfg.dataset.test_source, model_config_params=cfg.models, adapter_config_params=cfg.adapters, tokenizer=model.tokenizer, target_lang=lang_split, pred_path=cfg.dataset.pred_filepath)
            trainer.test(model, test_dataloader)
        merge_results(test_dataset, cfg.dataset.merge_filepath, cfg.dataset.pred_filepath, cfg.dataset.source_filepath)

if __name__ == "__main__":
    main()
