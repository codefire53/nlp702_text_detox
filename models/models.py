import lightning as L
from omegaconf import DictConfig
from factory.adapter_config_factory import create_adapter_config
from factory.adapter_model_factory import initialize_adapter_model, initialize_tokenizer
from transformers import AdamW, AutoTokenizer, AutoModelForSequenceClassification
import evaluate
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
import csv

class AdapterSeq2SeqModel(L.LightningModule):
    def __init__(self, model_config_params: DictConfig, adapter_config_params: DictConfig, optim_config_params:DictConfig = None, target_lang: str = None, pred_path: str = None):
        super().__init__()
        self.adapter_config = create_adapter_config(adapter_config_params)
        self.target_lang = target_lang
        if self.target_lang is not None and hasattr(model_config_params, "multilang"):
            model_config_params.multilang.target_id = self.target_lang
            model_config_params.multilang.source_id = self.target_lang
        self.model = initialize_adapter_model(model_config_params, adapter_config_params.name, self.adapter_config)
        self.tokenizer = initialize_tokenizer(self.model, model_config_params)
        self.optim_config_params = optim_config_params

        self.pred_path = pred_path
        self.test_preds = []
        

        # Load metrics
        self.bleu_metric = evaluate.load('sacrebleu')
        self.rouge_metric = evaluate.load('rouge')
        self.chrf_metric = evaluate.load('chrf')
        
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        self.log('val_loss', loss, prog_bar=True)

        # Generate predictions
        generated_tokens = self.model.generate(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        decoded_preds = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)
        
        # Remove -100 indices (labels for padding tokens)
        decoded_labels = [label.replace(self.tokenizer.pad_token, '') for label in decoded_labels]
        decoded_labels_bleu = [[label] for label in decoded_labels]
        
        # Update metrics
        self.bleu_metric.add_batch(predictions=decoded_preds, references=decoded_labels_bleu)
        self.rouge_metric.add_batch(predictions=decoded_preds, references=decoded_labels)
        self.chrf_metric.add_batch(predictions=decoded_preds, references=decoded_labels)

    def on_validation_epoch_end(self):
        # Compute metrics
        bleu_result = self.bleu_metric.compute()
        rouge_result = self.rouge_metric.compute()
        chrf_result = self.chrf_metric.compute(word_order=2)

        # Log metrics
        self.log_dict({'val_bleu': bleu_result['score'], 'val_rouge': rouge_result['rougeL'], 
                       'val_chrf++': chrf_result['score']}, prog_bar=True)


    def test_step(self, batch, batch_idx):
        # Generate predictions
        
        generated_tokens = self.model.generate(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], forced_bos_token_id=self.tokenizer.get_lang_id(self.target_lang))
        
        decoded_sentences = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        decoded_sentences = [sentence.strip() for sentence in decoded_sentences]
        self.test_preds.extend(decoded_sentences)

    def on_test_epoch_end(self):
        test_preds = '\n'.join(self.test_preds)
        path_prefix, path_suffix = self.pred_path.rsplit('.', 1)
        filepath = f"{path_prefix}_{self.target_lang}.{path_suffix}"
        with open(filepath, "w") as file:
            file.write(test_preds)
            file.write('\n')


    def configure_optimizers(self):
        return AdamW(self.model.parameters(), **self.optim_config_params)