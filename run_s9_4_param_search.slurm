#!/bin/bash
#SBATCH --array=1-84%14
#SBATCH --job-name=s_9_4_ML_LLM_param_search
#SBATCH -o slurm_param_search_logs/s_9_4_ML_LLM_param_search_%A_%a.log
#SBATCH -e slurm_param_search_logs/s_9_4_ML_LLM_param_search_%A_%a.log
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=96G
#SBATCH --partition=sapphire #bigmem
##SBATCH --qos=cpu_normal
##SBATCH --gres=gpu:2
#SBATCH --time=7-00:00:00
#SBATCH --nice=10000

set -euo pipefail

## Delete all previous log files
#rm -rf ./slurm_param_search_logs/*

echo "Loading anaconda"
module load Anaconda3/2024.02-1
eval "$(conda shell.bash hook)"

echo "Activate conda env"
conda activate scarches

# Specify the path to the config file
config="train_param_combinations.tsv"

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



python s9_4_ML_on_LLM_embeddings_param_search.py \
                    --dataset_name $dataset_name\
                    --data_inclusion_type $data_inclusion_type \
                    --model $model \
                    --period_end_day $period_end_day \
                    --overwrite_existing_params $overwrite_existing_params
                    #--cpu_cores 0-3