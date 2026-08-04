#!/bin/bash
#SBATCH --array=1-14%14
#SBATCH --job-name=s_8_4_lstm_training
#SBATCH -o slurm_lstm_train_logs/s_8_4_lstm_training_%A_%a.log
#SBATCH -e slurm_lstm_train_logs/s_8_4_lstm_training_%A_%a.log
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=24G
#SBATCH --partition=sapphire #bigmem
##SBATCH --qos=cpu_normal
##SBATCH --gres=gpu:2
#SBATCH --time=48:00:00
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
config="lstm_train_param_combinations.tsv"

#SLURM_ARRAY_TASK_ID=8

# Extract the parameters from the parameter configuration dataframe
period_end_day=$(awk -F'\t' -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$4==ArrayTaskID {print $2}' $config)
#data_inclusion_type=$(awk -F'\t' -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$4==ArrayTaskID {print $4}' $config)
model_complexity=$(awk -F'\t' -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$4==ArrayTaskID {print $3}' $config)
dataset_name=$(awk -F'\t' -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$4==ArrayTaskID {print $1}' $config)


#echo ${%a}
echo ${dataset_name}
echo ${model_complexity}
echo ${period_end_day}



/data/gpfs/projects/punim2121/anaconda3/envs/scarches/bin/python s8_4_LSTM_classification_training.py \
                    --dataset_name $dataset_name\
                    --model_complexity $model_complexity \
                    --period_end_day $period_end_day 