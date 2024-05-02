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
python evaluate_pred.py --prediction ./dataset/bloomz_pred_dev.json  -i ./dataset/source_dev.json -g ./multidetox_gt_dev.json -o ./eval_files/bloomz_dev.txt
```
Since we dont have ground truth for test set, alternatively we compare the chrf score between the source sentences their respective predictions since we hope that there will be larger overlaps between the ground truths and the source/toxic sentences. Thus for test set, you can run
```
python evaluate_pred.py --prediction ./dataset/bloomz_pred_test.json  -i ./dataset/source_test.json -g ./dataset/source_test.json -o ./eval_files/bloomz_test.txt
```