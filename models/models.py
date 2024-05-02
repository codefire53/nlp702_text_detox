import lightning as L
from omegaconf import DictConfig
from factory.adapter_config_factory import create_adapter_config
from factory.adapter_model_factory import initialize_adapter_model, initialize_tokenizer, initialize_model, add_adapter_fusion
from transformers import AdamW, AutoTokenizer, AutoModelForSequenceClassification
import evaluate
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
import csv
import json

import copy
from torch import nn

def freeze_all_params(model):
    for param in model.base_model.parameters():
        param.requires_grad = False


class LoRALayer(object):
    def __init__(self, scaler_coeff: float = 0.1):
        self.scaler_coeff = scaler_coeff
        self.use_lora = True

    def freeze_params(self, module):
        for param in module.parameters():
            param.requires_grad = False

class EmbeddingLoRA(LoRALayer, torch.nn.Module):
    def __init__(self, embedding, num_embeddings: int, downproj_coeff: int, embedding_dim: int, scaler_coeff: float = 0.1, use_bias: bool = True):
        torch.nn.Module.__init__(self)
        LoRALayer.__init__(self, scaler_coeff)

        self.num_embeddings = num_embeddings
        if downproj_coeff > num_embeddings:
            self.hidden_dim = num_embeddings
        else:
            self.hidden_dim = num_embeddings//downproj_coeff
        self.embedding_dim = embedding_dim

        self.embedding = embedding
        self.freeze_params(self.embedding)
        
        self.A = torch.nn.Embedding(self.num_embeddings, self.hidden_dim)
        self.B = torch.nn.Linear(self.hidden_dim, self.embedding_dim, bias=use_bias)
    
    def forward(self, x: torch.Tensor):
        hidden_state = self.embedding(x)

        if self.use_lora:
            res = self.B(self.A(x))
            hidden_state = hidden_state + (self.scaler_coeff/self.hidden_dim)*res
        return hidden_state
    

    def merge_weights(self):
        self.embedding.weight.data += (self.scaler_coeff/self.hidden_dim) * self.A.weight @ self.B.weight.T
        self.use_lora = False
        self.A = None
        self.B = None

    
class LinearLoRA(LoRALayer, torch.nn.Module):
    def __init__(self, linear, in_features: int, downproj_coeff: int, out_features: int, scaler_coeff: float = 0.1, use_bias: bool = True):
        torch.nn.Module.__init__(self)
        LoRALayer.__init__(self, scaler_coeff)

        self.in_features = in_features
        self.hidden_dim = in_features//downproj_coeff
        self.out_features = out_features

        self.linear = linear
        self.freeze_params(self.linear)
        
        self.A = torch.nn.Linear(self.in_features, self.hidden_dim, bias=use_bias)
        self.B = torch.nn.Linear(self.hidden_dim, self.out_features, bias=use_bias)
    
    def forward(self, x: torch.Tensor):
        hidden_state = self.linear(x)

        if self.use_lora:
            res = self.B(self.A(x))
            hidden_state = hidden_state + (self.scaler_coeff/self.hidden_dim)*res
        return hidden_state

    def merge_weights(self):
        self.linear.weight.data += (self.scaler_coeff/self.hidden_dim) * self.B.weight@ self.A.weight
        self.use_lora = False
        self.A = None
        self.B = None

def set_inference_mode(model):
    for _, module in model.named_modules():
        if isinstance(module, LoRALayer):
            print("Set inference mode")
            module.merge_weights()



def load_lora_target(filepath: str):
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data['lora_targets']
    

def copy_n_layers(n: int, layer: torch.nn.Module):
    return [copy.deepcopy(layer) for _ in range(n)]

class GateLayer(torch.nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_experts: int, top_k: int=3, threshold: float = 0.0):
        super(GateLayer, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.softmax = torch.nn.Softmax(dim=-1)
        self.layer1 = torch.nn.Linear(in_dim, hidden_dim)
        self.layer2 = torch.nn.Linear(hidden_dim, num_experts)
        self.threshold = threshold
        self.top_k = top_k

        self.total_assignments_so_far = torch.zeros(num_experts)

        self.noise_weight = torch.nn.Parameter(torch.zeros(in_dim, num_experts))

    def _get_trainable_noise(self, x: torch.Tensor):
        weighted_noise = x @ self.noise_weight
        return torch.randn_like(weighted_noise)*weighted_noise
    
    def forward(self, x: torch.Tensor):
        noise = self._get_trainable_noise(x)
        self.total_assignments_so_far = self.total_assignments_so_far.to(x.device)
        x = self.layer2(torch.nn.functional.relu(self.layer1(x)))
        x = x + noise
        expert_weights = self.softmax(x)
        step_total_assignments = torch.sum(expert_weights, dim=0)
        self.total_assignments_so_far = self.total_assignments_so_far + step_total_assignments
        mean_total_assignments = torch.mean(self.total_assignments_so_far)

        expert_mask = self.total_assignments_so_far.clone().to(x.device).detach()

        expert_mask[self.total_assignments_so_far - mean_total_assignments > self.threshold] = 0
        expert_mask[self.total_assignments_so_far - mean_total_assignments <= self.threshold] = 1

        expert_weights = expert_weights*expert_mask

        _, bottom_k_indices = expert_weights.topk(k=self.num_experts-self.top_k, dim=-1, largest=False)

        expert_weights = expert_weights.scatter(dim=-1, index=bottom_k_indices, value=-float('inf'))

        expert_weights = torch.nn.functional.softmax(expert_weights, dim=1)

        return expert_weights

class MoELayer(torch.nn.Module):
    def __init__(self, experts: list, gate_layer: GateLayer, is_sequence: bool = False):
        super(MoELayer, self).__init__()
        
        self.experts = torch.nn.ModuleList(experts)
        self.num_experts = len(self.experts)
        
        self.gate_layer = gate_layer
        self.is_sequence = is_sequence

        self.permutation_axes = (1,2,0,3) if self.is_sequence else(1,0,2)

    def forward(self, x: torch.Tensor):
        gates_proba = self.gate_layer(x) # batch x seq_length x num_experts
        expert_outputs = torch.stack([expert(x) for expert in self.experts])
        expert_outputs = expert_outputs.permute(*self.permutation_axes) # batch_size, seq_length, num_experts, dim
        gates_proba = gates_proba.unsqueeze(-1)
        
        output = torch.sum(gates_proba*expert_outputs, dim=-2)
        return output
    

def inject_molora(model, target_layers, adapter_params: DictConfig):
    for target_layer in target_layers:
        target_layer, is_sequence = target_layer[0], target_layer[1]
        splitted_target_layer = target_layer.split(".")
        parents = splitted_target_layer[:-1]
        child = splitted_target_layer[-1]
        parent_module = model

        for parent in parents:
            parent_module = getattr(parent_module, parent)
        
        old_child = getattr(parent_module, child)
        new_child = LinearLoRA(old_child, old_child.in_features, adapter_params.lora.r, old_child.out_features, adapter_params.lora.lora_alpha, adapter_params.lora.use_bias)
        experts = copy_n_layers(adapter_params.gate.num_experts, new_child)
        gate_layer = GateLayer(old_child.in_features, adapter_params.gate.hidden_dim, adapter_params.gate.num_experts, adapter_params.gate.top_k
        , adapter_params.gate.threshold)
        moe_layer = MoELayer(experts, gate_layer, is_sequence)
        print(f"Replacing {target_layer}")
        setattr(parent_module, child, moe_layer)
    print("Model summary:")
    print(model)
    

def load_molora_target(filepath: str):
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data['moe_target']

class BaseLightningModule(L.LightningModule):
    def __init__(self, model_config_params: DictConfig, optim_config_params:DictConfig = None, target_lang: str = None, pred_path: str = None):
        super().__init__()
        self.target_lang = target_lang
        if self.target_lang is not None and hasattr(model_config_params, "multilang"):
            model_config_params.multilang.target_id = self.target_lang
            model_config_params.multilang.source_id = self.target_lang
        self.model = initialize_model(model_config_params)
        self.tokenizer = initialize_tokenizer(self.model, model_config_params)
        self.optim_config_params = optim_config_params

        self.pred_path = pred_path
        self.test_preds = []
        

        # Load metrics
        self.bleu_metric = evaluate.load('sacrebleu')
        self.rouge_metric = evaluate.load('rouge')
        self.chrf_metric = evaluate.load('chrf')

        self.chrf_scores = []
        self.polite_scores = []
        self.sim_scores = []
        
        
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
        #Compute metrics
        bleu_result = self.bleu_metric.compute()
        rouge_result = self.rouge_metric.compute()
        chrf_result = self.chrf_metric.compute(word_order=2)

        # Log metrics
        self.log_dict({'val_bleu': bleu_result['score'], 'val_rouge': rouge_result['rougeL'], 
                      'val_chrf++': chrf_result['score']}, prog_bar=True)


    def translate_sentence(self, src_sent, lang):
        self.tokenizer.src_lang = lang
        self.tokenizer.set_src_lang_special_tokens(self.tokenizer.src_lang)
        self.tokenizer.tgt_lang = lang
        encoded_text = self.tokenizer(src_sent, return_tensors="pt").to('cuda')
        translated_text = self.model.generate(**encoded_text, forced_bos_token_id=self.tokenizer.get_lang_id(lang))
        translated_text = self.tokenizer.batch_decode(translated_text, skip_special_tokens=True)
        return translated_text[0].strip()


    def configure_optimizers(self):
        return AdamW(self.model.parameters(), **self.optim_config_params)



class AdapterSeq2SeqModel(BaseLightningModule):
    def __init__(self, model_config_params: DictConfig, adapter_config_params: DictConfig, optim_config_params:DictConfig = None, target_lang: str = None, pred_path: str = None):
        super().__init__(model_config_params, optim_config_params, target_lang, pred_path)
        self.adapter_config = create_adapter_config(adapter_config_params)
        self.model = initialize_adapter_model(self.model, self.adapter_config)
        self.use_lora = True


class MoLoRAModel(BaseLightningModule):
    def __init__(self, model_config_params: DictConfig, adapter_config_params: DictConfig, optim_config_params:DictConfig = None, target_lang: str = None, pred_path: str = None):
        super().__init__(model_config_params, optim_config_params, target_lang, pred_path)
        moe_targets = load_molora_target(adapter_config_params.targets_file)
        freeze_all_params(self.model)
        inject_molora(self.model, moe_targets, adapter_config_params)


    def set_inference_mode(self):
        if self.use_lora:
            print("Merging all lora weights before inference")
            set_inference_mode(self.model)


class AdapterFusionModel(BaseLightningModule):
    def __init__(self, model_config_params: DictConfig, adapter_config_params: DictConfig, optim_config_params:DictConfig = None, target_lang: str = None, pred_path: str = None):
        super().__init__(model_config_params, optim_config_params, target_lang, pred_path)
        add_adapter_fusion(self.model, adapter_config_params)
