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
