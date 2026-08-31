#!/bin/bash
#SBATCH --array=1-14%14
##SBATCH --array=3,4,11,12
##SBATCH --array=1,2,5,6,7,8,9,10,13,14
#SBATCH --job-name=s_8_4_lstm_param_search
#SBATCH -o slurm_lstm_param_search_logs/s_8_4_lstm_param_search_%A_%a.log
#SBATCH -e slurm_lstm_param_search_logs/s_8_4_lstm_param_search_%A_%a.log
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=24G
#SBATCH --partition=sapphire #bigmem
##SBATCH --qos=cpu_normal
##SBATCH --gres=gpu:1
#SBATCH --time=7-00:00:00
#SBATCH --nice=10000

set -euo pipefail

## Delete all previous log files
#rm -rf ./slurm_param_search_logs/*



echo "Loading anaconda"
module load Anaconda3/2024.02-1
eval "$(conda shell.bash hook)"

#echo "Activate conda env"
#conda activate scarches

## A specific python within the desired environment (here: scarches)is executing the python file ==> this needs to be adjusted if environment or HPC changes
echo "Activating conda env"
conda activate "${CONDA_ENV_PATH:?Set CONDA_ENV_PATH to your scarches environment path, e.g. export CONDA_ENV_PATH=/path/to/anaconda3/envs/scarches}"
# Add the environments's python bin path to the first place in $PATH --> this way the environment's own python will be run when using "python" command
export PATH="${CONDA_PREFIX}/bin:${PATH}"

echo "Python being used:"
which python

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


python s8_4_LSTM_classification_param_search.py \
                    --dataset_name $dataset_name\
                    --model_complexity $model_complexity \
                    --period_end_day $period_end_day 
