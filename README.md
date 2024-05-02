# Setup
Setup the environment
```
conda create -n multidetox python=3.10
```
Then install all required dependencies.
```
pip install -r requirements.txt
```

# How to run the PEFT methods experiment
This project uses `hydra` to manage the experiment configuration. Therefore you need to adjust the `adapter_experiment_config.yaml` file. The important parts to adjust are `loggers`, `dataset` (to change which datasets should be used for training), `checkpoint`, `do_train` (training toggle), `do_test` (test set inference toggle), `in_filepath`, `out_filepath`, `out_src_eval_filepath`, `out_pred_eval_filepath` , `out_tgt_eval_filepath`, and `checkpoint_file`. Once you have set the configuration, the next step is to run 
```
python adapter_experiment.py
```


# Evaluation
To do evaluation on validation set (please just differentiate the prediction and o argument here)
```
python evaluate_pred.py --prediction ./dataset/bloomz_pred_dev.json  -l ./dataset/lang_dev.json -i ./dataset/source_dev.json -g ./multidetox_gt_dev.json -o ./eval_files/bloomz_dev.txt
```

# mT5
We adapted the notebook for fine-tuning text2text transformer models using HuggingFace trainer from [here](https://github.com/UBC-NLP/araT5/blob/main/examples/Fine_tuning_AraT5.ipynb). To fine-tune the model, run the following command:
```
!python mt5/train_mt5.py \
        --learning_rate 5e-5 \
        --max_target_length 128 --max_source_length 128 \
        --per_device_train_batch_size 8 --per_device_eval_batch_size 8 \
        --model_name_or_path "google/mt5-base" \
        --output_dir "mt5/mT5_FT" --overwrite_output_dir \
        --num_train_epochs 22 \
        --train_file "dataset/train_mt5.tsv" \
        --validation_file "dataset/dev_mt5.tsv" \
        --task "detoxify" --text_column "toxic" --summary_column "neutral" \
        --load_best_model_at_end --metric_for_best_model "chrf" --greater_is_better True --evaluation_strategy steps --logging_strategy steps --predict_with_generate \
        --do_train --do_eval > log.log 2>&1
```

To run inference on the test set, you can run the following command:
```
python mt5/mt5_inference.py
```

# Checkpoints
To get checkpoints for our experiments you can download them from [here](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/mahardika_ihsani_mbzuai_ac_ae/Ek_E5ueFodhEjmq4djyraUgB6s3ngC4F7328JEXIXjn-dA?e=hAsNuY) (for PEFT and mT5 approaches).
