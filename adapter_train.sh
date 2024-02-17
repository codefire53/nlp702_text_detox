#!/bin/bash

#SBATCH --job-name=unipelt_paradetox_mt5 # Job name
#SBATCH --error=./logs/%j%x.err # error file
#SBATCH --output=./logs/%j%x.out # output log file
#SBATCH --time=24:00:00 # 10 hours of wall time
#SBATCH --nodes=1  # 1 GPU node
#SBATCH --mem=16000 # 16 GB of RAM
#SBATCH --nodelist=ws-l6-007


echo "Starting......................"
python adapter_train.py models=mt5 adapters=unipelt dataset=paradetox tokenizers=mt5 run_name=mt5-base_unipelt_paradetox output_dir=./outputs/mt5-base_unipelt_paradetox/checkpoints logging_dir=./outputs/mt5-base_unipelt_paradetox/logs
