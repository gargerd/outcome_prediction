#!/bin/bash
#SBATCH --array=1-112%112
#SBATCH --job-name=s_9_9_raw_survival_param_search
#SBATCH -o slurm_raw_survival_param_search_logs/s_9_9_raw_survival_param_search%A_%a.log
#SBATCH -e slurm_raw_survival_param_search_logs/s_9_9_raw_survival_param_search%A_%a.log
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=24G
#SBATCH --partition=sapphire #bigmem
##SBATCH --qos=cpu_normal
##SBATCH --gres=gpu:2
#SBATCH --time=0-12:00:00
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
config="raw_survival_train_param_combinations.tsv"

#SLURM_ARRAY_TASK_ID=1

# Extract the parameters from the parameter configuration dataframe
period_end_day=$(awk -F'\t' -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$6==ArrayTaskID {print $3}' $config)
training_data_type=$(awk -F'\t' -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$6==ArrayTaskID {print $4}' $config)
model=$(awk -F'\t' -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$6==ArrayTaskID {print $2}' $config)
dataset_name=$(awk -F'\t' -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$6==ArrayTaskID {print $1}' $config)
ther_arm_duration=$(awk -F'\t' -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$6==ArrayTaskID {print $5}' $config)

## Set string if to overwrite existing parameter search results
overwrite_existing_params='True'

#echo ${%a}
echo ${dataset_name}
echo ${training_data_type}
echo ${period_end_day}
echo ${model}
echo ${ther_arm_duration}



/data/gpfs/projects/punim2121/anaconda3/envs/scarches/bin/python  s9_9_baseline_survival_models_param_search.py \
                                                                    --dataset_name $dataset_name\
                                                                    --training_data_type $training_data_type \
                                                                    --model $model \
                                                                    --period_end_day $period_end_day \
                                                                    --ther_arm_duration $ther_arm_duration
      