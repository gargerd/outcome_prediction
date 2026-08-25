#!/bin/bash
#SBATCH --array=1-14%14
#SBATCH --job-name=s_9_3_raw_data_test
#SBATCH -o slurm_raw_data_testing_logs/s_9_3_raw_data_test_%A_%a.log
#SBATCH -e slurm_raw_data_testing_logs/s_9_3_raw_data_test_%A_%a.log
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=12G
#SBATCH --partition=sapphire #bigmem
##SBATCH --qos=cpu_normal
##SBATCH --gres=gpu:2
#SBATCH --time=0-06:00:00
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
config="raw_data_test_param_combinations.tsv"

#SLURM_ARRAY_TASK_ID=1

# Extract the parameters from the parameter configuration dataframe
#period_end_day=$(awk -F'\t' -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$5==ArrayTaskID {print $3}' $config)
#training_data_type=$(awk -F'\t' -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$5==ArrayTaskID {print $4}' $config)
#model=$(awk -F'\t' -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$5==ArrayTaskID {print $2}' $config)
#dataset_name=$(awk -F'\t' -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$5==ArrayTaskID {print $1}' $config)


# Get unique dataset_name and training_data_type (1 per ArrayTaskID)
read dataset_name training_data_type <<< $(awk -F'\t' -v ArrayTaskID=$SLURM_ARRAY_TASK_ID \
    '$5==ArrayTaskID {print $1, $4; exit}' $config)

# Get all models and period_end_days as arrays
#models=($(awk -F'\t' -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$5==ArrayTaskID {print $2}' $config))
#period_end_days=($(awk -F'\t' -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$5==ArrayTaskID {print $3}' $config))

#models=($(awk -F'\t' -v ArrayTaskID=1 '$5==1 {print $2}' $config))
#period_end_days=($(awk -F'\t' -v ArrayTaskID=1 '$5==1 {print $3}' $config))

models=$(awk -F'\t' -v id="$SLURM_ARRAY_TASK_ID" '$5==id {print $2}' "$config" | sort -u | paste -sd,)
period_end_days=$(awk -F'\t' -v id="$SLURM_ARRAY_TASK_ID" '$5==id {print $3}' "$config" | sort -u | paste -sd,)


# Convert arrays to comma-separated strings
models_list=$(IFS=','; echo "${models[*]}")
periods_list=$(IFS=','; echo "${period_end_days[*]}")

## Set string if to overwrite existing parameter search results
overwrite_existing_params='True'

#echo ${%a}
#echo ${dataset_name}
#echo ${training_data_type}
#echo ${period_end_day}
echo ${periods_list}
#echo ${model}
echo ${models_list}


## A specific python within the desired environment (here: scarches)is executing the python file ==> this needs to be adjusted if environment or HPC changes
/data/gpfs/projects/punim2121/anaconda3/envs/scarches/bin/python s9_3_baseline_ML_models_testing.py \
                                                                --dataset_name $dataset_name\
                                                                --training_data_type $training_data_type \
                                                                --model "$models_list" \
                                                                --period_end_day "$periods_list"
                                            
                                                                #--model $model \
                                                                #--period_end_day $period_end_day 
      