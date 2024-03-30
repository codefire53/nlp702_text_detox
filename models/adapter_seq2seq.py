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
        # self.bert_tokenizer = AutoTokenizer.from_pretrained("textdetox/xlmr-base-toxicity-classifier")
        # self.bert_model = AutoModelForSequenceClassification.from_pretrained("textdetox/xlmr-base-toxicity-classifier", num_labels=2).to('cuda')
        # self.bert_model.eval()
        # self.sts_model = SentenceTransformer('sentence-transformers/LaBSE')

        # self.chrf_scores = []
        # self.polite_scores = []
        # self.sim_scores = []
        
        
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

    # def _eval_polite_score(self, sentences):
    #     polite_label = 0
    #     inputs = self.bert_tokenizer(
    #         *sentences,
    #         return_tensors="pt",
    #         padding=True,
    #         truncation=True,
    #         max_length=512,
    #     ).to(self.bert_model.device)
    #     with torch.no_grad():
    #         try:
    #             logits = model(**inputs).logits
    #             preds = torch.softmax(logits, -1)[:,polite_label]
    #             preds = polite_scores.cpu().numpy()
    #         except:
    #             preds = [0] * len(inputs)
    #     print(f"politeness: {preds}")
    #     self.polite_scores.extend(preds)

    # def _compute_cosine_sim(self, embds1, embds2):
    #     sim_matrix = np.dot(embds1, embds2.T)
    #     norms1 = np.linalg.norm(embds1, axis=1)
    #     norms2 = np.linalg.norm(embds2, axis=1)
    #     score = sim_matrix / (np.outer(norms1, norms2) + 1e-9)

    # def _eval_sim_score(self, source_sentences, detox_sentences):
    #     source_len = len(source_sentences)
    #     detox_len = len(detox_sentences)
    #     all_texts = detox_sentences + source_sentences
    #     embds = self.sts_model.encode(all_texts)
    #     detox_embds = embds[:detox_len] 
    #     source_embds = embds[detox_len:]
    #     scores = []
    #     sim_scores = self._compute_cosine_sim(detox_embds, source_embds)
    #     for i in range(detox_len):
    #         sim_score = max(sim_scores[i,i+detox_len], 0)
    #         scores.append(sim_score)
    #     print(f"similarity: {scores}")
    #     self.sim_scores.extend(scores)
    
    # def _eval_chrf_score(self, source_sentences, detox_sentences):
    #     scores = []
    #     for src, detox in zip(source_sentences, detox_sentences):
    #         chrf_score = self.chrf_metric.compute(predictions=[detox], references=[src], word_order=2)
    #         chrf_score = chrf_score['score']/100
    #         scores.append(chrf_score)
    #     print(f"chrf: {scores}")
    #     self.chrf_scores(scores)

    def test_step(self, batch, batch_idx):
        # Generate predictions
        
        generated_tokens = self.model.generate(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], forced_bos_token_id=self.tokenizer.get_lang_id(self.target_lang))
        
        decoded_sentences = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        decoded_sentences = [sentence.strip() for sentence in decoded_sentences]
        self.test_preds.extend(decoded_sentences)

        # self._eval_polite_score([decoded_sentences])
        # self._eval_chrf_score(original_sentences, detox_sentences)
        # self._eval_sim_score(original_sentences, detox_sentences)

    def on_test_epoch_end(self):
        # agg_scores = np.array(self.chrf_scores)*np.array(self.sim_scores)*np.array(self.polite_scores)
        # overall_total = sum(agg_scores)/len(agg_scores)
        # overall_chrf = sum(self.chrf_scores)/len(self.chrf_scores)
        # overall_sim = sum(self.sim_scores)/len(self.sim_scores)
        # overall_polite = sum(self.polite_scores)/len(self.polite_scores)


        # Log metrics
        # self.log_dict({'test_politeness': overall_polite, 'test_similarity': overall_sim, 'test_total': overall_total, 'test_chrf': overall_chrf}, prog_bar=True)
        
        test_preds = '\n'.join(self.test_preds)
        path_prefix, path_suffix = self.pred_path.rsplit('.', 1)
        filepath = f"{path_prefix}_{self.target_lang}.{path_suffix}"
        with open(filepath, "w") as file:
            file.write(test_preds)
            file.write('\n')


    def configure_optimizers(self):
        return AdamW(self.model.parameters(), **self.optim_config_params)