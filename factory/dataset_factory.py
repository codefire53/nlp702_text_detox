from transformers import AutoTokenizer
from datasets import load_dataset, Dataset
from omegaconf import DictConfig

def tokenize_function(examples: list, tokenizer: AutoTokenizer, tokenizer_config_params: DictConfig, source: str, target: str):
        model_inputs = tokenizer(examples[source], **tokenizer_config_params)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples[target], **tokenizer_config_params)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

def load_detox_train_sets(dataset_config_params: DictConfig, tokenizer_config_params: DictConfig, tokenizer: AutoTokenizer) -> Dataset:
    dataset = load_dataset(dataset_config_params.dataset_name)
    if dataset_config_params.use_split:
        if dataset_config_params.use_json:
            data_files = {'train': dataset_config_params.train_dataset_file, 'validation': dataset_config_params.val_dataset_file}
            dataset = load_dataset('json', data_files=data_files)
        else:     
            dataset = dataset['train'].train_test_split(test_size=dataset_config_params.test_size)
            dataset['validation'] = dataset['test']
            dataset['train'].to_json(dataset_config_params.train_dataset_file)
            dataset['validation'].to_json(dataset_config_params.val_dataset_file)
    tokenized_train = dataset['train'].map(lambda examples: tokenize_function(examples, tokenizer, tokenizer_config_params, dataset_config_params.source, dataset_config_params.target))
    tokenized_val = dataset['validation'].map(lambda examples: tokenize_function(examples, tokenizer, tokenizer_config_params, dataset_config_params.source, dataset_config_params.target))
    return tokenized_train, tokenized_val