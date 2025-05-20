#!/bin/bash

set -euo pipefail

# Specify the path to the config file
config="train_param_combinations.tsv"

SLURM_ARRAY_TASK_ID=1

# Extract the parameters from the parameter configuration dataframe
period_end_day=$(awk -F'\t' -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$5==ArrayTaskID {print $3}' $config)
data_inclusion_type=$(awk -F'\t' -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$5==ArrayTaskID {print $4}' $config)
model=$(awk -F'\t' -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$5==ArrayTaskID {print $2}' $config)
dataset_name=$(awk -F'\t' -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$5==ArrayTaskID {print $1}' $config)

## Set string if to overwrite existing parameter search results
overwrite_existing_params='True'

#echo ${%a}
echo ${dataset_name}
echo ${data_inclusion_type}
echo ${period_end_day}
echo ${model}