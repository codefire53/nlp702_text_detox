from transformers import AutoTokenizer
from datasets import load_dataset
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from factory.collator_factory import initialize_collator_module

def tokenize_function(examples: list, tokenizer: AutoTokenizer, tokenizer_config_params: DictConfig, source: str, target: str = None):
    model_inputs_str = [example.strip() for example in examples[source]]
    model_inputs = tokenizer(model_inputs_str, **tokenizer_config_params)
    if target is not None:
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples[target], **tokenizer_config_params)
        model_inputs["labels"] = labels["input_ids"]
    
    return model_inputs

def load_dataset_from_file_or_name(filepath_or_name: str, split: str = "train"):
    if filepath_or_name.endswith(".json"):
          return load_dataset("json", data_files=filepath_or_name, split=split)
    return load_dataset(filepath_or_name, split=split)

def load_detox_sets(dataset_config_params: DictConfig, tokenizer_config_params: DictConfig, tokenizer: AutoTokenizer, collator_params: DictConfig, dataloader_params: DictConfig) -> [DataLoader, DataLoader, DataLoader]:
    if dataset_config_params.use_split:
        train_dataset = load_dataset_from_file_or_name(dataset_config_params.train_dataset_name_or_file)
        train_dataset = train_dataset.train_test_split(test_size=dataset_config_params.val_size)
        val_dataset = train_dataset['test']
        train_dataset = train_dataset['train']
        train_dataset.to_json(dataset_config_params.train_dataset_file)
        val_dataset.to_json(dataset_config_params.val_dataset_file)
    else:
        train_dataset = load_dataset_from_file_or_name(dataset_config_params.train_dataset_name_or_file, dataset_config_params.train_split)
        val_dataset = load_dataset_from_file_or_name(dataset_config_params.val_dataset_name_or_file, dataset_config_params.val_split)


    # tokenize the datasets
    tokenized_train = train_dataset.map(lambda examples: tokenize_function(examples, tokenizer, tokenizer_config_params, dataset_config_params.train_source, dataset_config_params.train_target), batched=True)
    tokenized_val = val_dataset.map(lambda examples: tokenize_function(examples, tokenizer, tokenizer_config_params, dataset_config_params.val_source, dataset_config_params.val_target), batched=True)
    
    # remove unnecessart cols 
    tokenized_train = tokenized_train.remove_columns([dataset_config_params.train_source, dataset_config_params.train_target])
    tokenized_val = tokenized_val.remove_columns([dataset_config_params.val_source, dataset_config_params.val_target])
    collator_fn = initialize_collator_module(tokenizer, collator_params)

    # create dataloader
    tokenized_train = DataLoader(tokenized_train, shuffle=True, collate_fn=collator_fn, **dataloader_params)
    tokenized_val = DataLoader(tokenized_val, shuffle=False, collate_fn=collator_fn, **dataloader_params)

    return tokenized_train, tokenized_val

def load_test_dataset(dataset_config_params: DictConfig):
    return load_dataset(dataset_config_params.test_dataset_name_or_file)

def load_test_dataloader(dataset, dataset_config_params: DictConfig, tokenizer_config_params: DictConfig, tokenizer: AutoTokenizer, collator_params: DictConfig, dataloader_params: DictConfig, split: str):
    toxic_comments = [sentence.strip() for sentence in dataset[dataset_config_params.test_source]]
    toxic_comments = "\n".join(toxic_comments)
    path_prefix, path_suffix =  dataset_config_params.source_filepath.rsplit('.', 1)
    filepath = f"{path_prefix}_{split}.{path_suffix}"
    with open(filepath, 'w') as f:
        f.write(toxic_comments)
        f.write('\n')

    tokenized_test = dataset.map(lambda examples: tokenize_function(examples, tokenizer, tokenizer_config_params, dataset_config_params.test_source, None), batched=True)
    tokenized_test = tokenized_test.remove_columns([dataset_config_params.test_source])
    collator_fn = initialize_collator_module(tokenizer, collator_params)
    test_dl = DataLoader(tokenized_test, shuffle=False, collate_fn=collator_fn, **dataloader_params)

    return test_dl

