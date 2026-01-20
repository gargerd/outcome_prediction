#!/bin/bash
##SBATCH --array=1-84%20
#SBATCH --job-name=s_9_2_extract_LLM_embeddings
#SBATCH -o slurm_llm_embedding_logs/s_9_2_extract_LLM_embeddings_%A_%a.log
#SBATCH -e slurm_llm_embedding_logs/s_9_2_extract_LLM_embeddings_%A_%a.log
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=32G
#SBATCH --partition=gpu-h100
##SBATCH --qos=cpu_normal
#SBATCH --gres=gpu:1
#SBATCH --time=0-01:00:00
#SBATCH --nice=10000

set -euo pipefail


echo "Loading anaconda"
module load Anaconda3/2024.02-1
eval "$(conda shell.bash hook)"

echo "Activate conda env"
conda activate scarches

#echo $(conda info)


## Specify which days and inclusion types should be embedded

dataset_name='tb21_22_2984_pats_22_vars_result_at_end_of_treatment'
#dataset_name='tb21_22_2984_pats_22_vars_result_at_end_of_treatment_wo_dr_reg'

#data_inclusion_type='baseline_last_day baseline_vars all_days'
#pool_methods='mean_pooling attention_pooling'
pool_methods='mean_pooling'
#period_end_day='baseline 31 62 93 125 160 all'


data_inclusion_type='baseline_last_day baseline_vars'
period_end_day='baseline'



#echo ${%a}
echo ${data_inclusion_type}
echo ${period_end_day}


/data/gpfs/projects/punim2121/anaconda3/envs/scarches/bin/python  s9_2_LLM_extract_embedding.py \
                                                                --dataset_name $dataset_name \
                                                                --data_inclusion_type $data_inclusion_type \
                                                                --pool_methods $pool_methods \
                                                                --period_end_day $period_end_day \

