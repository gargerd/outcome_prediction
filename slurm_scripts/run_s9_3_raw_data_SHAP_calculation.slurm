#!/bin/bash
#SBATCH --array=1-56%56
##SBATCH --array=24,25,26,27,54,55,56,57,58
#SBATCH --job-name=s_9_3_raw_data_SHAP
#SBATCH -o slurm_raw_data_SHAP_logs/s_9_3_raw_data_SHAP_%A_%a.log
#SBATCH -e slurm_raw_data_SHAP_logs/s_9_3_raw_data_SHAP_%A_%a.log
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --partition=sapphire #bigmem
##SBATCH --qos=cpu_normal
##SBATCH --gres=gpu:2
#SBATCH --time=0-08:00:00
#SBATCH --nice=10000

set -euo pipefail

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
config="raw_data_train_param_combinations.tsv"

#SLURM_ARRAY_TASK_ID=1

# Extract the parameters from the parameter configuration dataframe
period_end_day=$(awk -F'\t' -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$5==ArrayTaskID {print $3}' $config)
training_data_type=$(awk -F'\t' -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$5==ArrayTaskID {print $4}' $config)
model=$(awk -F'\t' -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$5==ArrayTaskID {print $2}' $config)
dataset_name=$(awk -F'\t' -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$5==ArrayTaskID {print $1}' $config)

## Set string if to overwrite existing parameter search results
overwrite_existing_params='True'

#echo ${%a}
echo ${dataset_name}
echo ${training_data_type}
echo ${period_end_day}
echo ${model}



python s9_3_baseline_ML_models_SHAP_calculation.py \
                                                                --dataset_name $dataset_name\
                                                                --training_data_type $training_data_type \
                                                                --model $model \
                                                                --period_end_day $period_end_day 
      