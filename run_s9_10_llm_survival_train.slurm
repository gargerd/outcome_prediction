#!/bin/bash
#SBATCH --array=1-84%84
##SBATCH --array=46,48,49,50,51,52,56,57
#SBATCH --job-name=s_9_10_LLM_survival_train
#SBATCH -o slurm_llm_survival_train_logs/s_9_10_LLM_survival_train_%A_%a.log
#SBATCH -e slurm_llm_survival_train_logs/s_9_10_LLM_survival_train_%A_%a.log
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=24G
#SBATCH --partition=sapphire #bigmem
##SBATCH --qos=cpu_normal
##SBATCH --gres=gpu:2
#SBATCH --time=1-00:00:00
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
config="llm_survival_train_param_combinations.tsv"

#SLURM_ARRAY_TASK_ID=1

# Extract the parameters from the parameter configuration dataframe
period_end_day=$(awk -F'\t' -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$8==ArrayTaskID {print $3}' $config)
data_inclusion_type=$(awk -F'\t' -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$8==ArrayTaskID {print $4}' $config)
model=$(awk -F'\t' -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$8==ArrayTaskID {print $2}' $config)
dataset_name=$(awk -F'\t' -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$8==ArrayTaskID {print $1}' $config)
pool_method=$(awk -F'\t' -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$8==ArrayTaskID {print $5}' $config)
ther_arm_duration=$(awk -F'\t' -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$8==ArrayTaskID {print $6}' $config)
time_origin=$(awk -F'\t' -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$8==ArrayTaskID {print $7}' $config)

## Set string if to overwrite existing parameter search results
overwrite_existing_params='True'

#echo ${%a}
echo ${dataset_name}
echo ${data_inclusion_type}
echo ${period_end_day}
echo ${model}
echo ${pool_method}
echo ${ther_arm_duration}
echo ${time_origin}



/data/gpfs/projects/punim2121/anaconda3/envs/scarches/bin/python  s9_10_survival_on_LLM_embeddings_training.py \
                                                            --dataset_name $dataset_name\
                                                            --data_inclusion_type $data_inclusion_type \
                                                            --model $model \
                                                            --period_end_day $period_end_day \
                                                            --pool_method $pool_method \
                                                            --ther_arm_duration $ther_arm_duration \
                                                            --time_origin $time_origin
                                                     