#!/bin/bash
##SBATCH --array=1-84%20
#SBATCH --job-name=s_9_12_ML_LLM_survival_integrated_gradients
#SBATCH -o slurm_llm_survival_IG_logs/s_9_12_ML_LLM_integrated_gradients_%A_%a.log
#SBATCH -e slurm_llm_survival_IG_logs/s_9_12_ML_LLM_integrated_gradients_%A_%a.log
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=16G
#SBATCH --partition=gpu-h100
##SBATCH --qos=cpu_normal
#SBATCH --gres=gpu:2
#SBATCH --time=1-12:00:00
#SBATCH --nice=10000

set -euo pipefail


echo "Loading anaconda"
module load Anaconda3/2024.02-1
eval "$(conda shell.bash hook)"

echo "Activate conda env"
conda activate scarches

#echo $(conda info)


## Specify which model should be run with IG
dataset_name='tb21_22_2984_pats_22_vars_relapse'
#dataset_name='tb21_22_2984_pats_22_vars_relapse_without_dr_reg'
data_inclusion_type='all_days'
model='CoxnetSurvival'
period_end_day='160' #'all'
#period_end_day='baseline 31 62 93 125 160 all'
#data_inclusion_type='baseline_last_day'
internal_batch_size='1'
ther_arm_duration='6-month'

#dataset_name='tb21_22_2984_pats_22_vars_result_at_end_of_treatment'
#data_inclusion_type='baseline_last_day'
#model='LogisticRegression'
#period_end_day='baseline 31 62 93 125 160'
#internal_batch_size='4'

pool_method='mean_pooling'


#echo ${%a}
echo ${dataset_name}
echo ${data_inclusion_type}
echo ${period_end_day}
echo ${model}
echo ${pool_method}
echo ${ther_arm_duration}


/data/gpfs/projects/punim2121/anaconda3/envs/scarches/bin/python  s9_12_LLM_survival_integrated_gradients.py \
                                                                --dataset_name $dataset_name\
                                                                --data_inclusion_type $data_inclusion_type \
                                                                --model $model \
                                                                --period_end_day $period_end_day \
                                                                --internal_batch_size $internal_batch_size \
                                                                --pool_method $pool_method \
                                                                --ther_arm_duration $ther_arm_duration

                    #--cpu_cores 0-3
