#!/bin/bash
#SBATCH --array=22-42%42
#SBATCH --job-name=s_9_4_ML_LLM_training
#SBATCH -o slurm_llm_training_logs/s_9_4_ML_LLM_training_%A_%a.log
#SBATCH -e slurm_llm_training_logs/s_9_4_ML_LLM_training_%A_%a.log
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=24G
#SBATCH --partition=sapphire
##SBATCH --qos=cpu_normal
##SBATCH --gres=gpu:2
#SBATCH --time=1-00:00:00
#SBATCH --nice=10000

set -euo pipefail


echo "Loading anaconda"
module load Anaconda3/2024.02-1
eval "$(conda shell.bash hook)"

echo "Activate conda env"
conda activate scarches

# Specify the path to the config file
config="train_param_combinations.tsv"

#SLURM_ARRAY_TASK_ID=42

# Extract the parameters from the parameter configuration dataframe
period_end_day=$(awk -F'\t' -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$6==ArrayTaskID {print $3}' $config)
data_inclusion_type=$(awk -F'\t' -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$6==ArrayTaskID {print $4}' $config)
model=$(awk -F'\t' -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$6==ArrayTaskID {print $2}' $config)
dataset_name=$(awk -F'\t' -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$6==ArrayTaskID {print $1}' $config)
pool_method=$(awk -F'\t' -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$6==ArrayTaskID {print $5}' $config)


#echo ${%a}
echo ${dataset_name}
echo ${data_inclusion_type}
echo ${period_end_day}
echo ${model}



/data/gpfs/projects/punim2121/anaconda3/envs/scarches/bin/python s9_4_ML_on_LLM_embeddings_training.py \
                                                                --dataset_name $dataset_name\
                                                                --data_inclusion_type $data_inclusion_type \
                                                                --model $model \
                                                                --period_end_day $period_end_day \
                                                                --pool_method $pool_method
                                            
                                                                #--cpu_cores 0-3
