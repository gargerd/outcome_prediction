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



######## ======================================== GENERAL INFORMATION ========================================

#### ======= Specify which model should be run with IG ======= 
## - Experimental setup (here called as dataset_name) for EOT outcome prediction used in paper: 
#    dataset_name='tb21_22_2984_pats_22_vars_result_at_end_of_treatment_without_dr_reg'
## - Experimental setup (here called as dataset_name) for relapse prediction used in paper: 
#    dataset_name='tb21_22_2984_pats_22_vars_result_at_end_of_treatment_without_dr_reg'

#dataset_name='tb21_22_2984_pats_22_vars_relapse'
#dataset_name='tb21_22_2984_pats_22_vars_relapse_without_dr_reg'
#dataset_name='tb21_22_2984_pats_22_vars_relapse_ext_pats'
dataset_name='tb21_22_2984_pats_22_vars_relapse_without_dr_reg_ext_pats'

### =======  Specify the data input approach ======= 
## Called as data_inclusion_type variable here
## Possible values: 'all days': all variables - all visits in period
##                  'baseline_vars': common variables - all visits in period
##                  'baseline_last_day': common variables - visit at end of period


### =======  Specify the time period ======= 
## Possible values: period_end_day='baseline 31 62 93 125 160 all'
#. ==> this value is read as a list in python 
#.     i.e. python list: period_end_day=['baseline',31,62,93,125,160,'all'], is equal to period_end_day='baseline 31 62 93 125 160 all';
#.     the integers in the bash list are strings here, (31-160), but they are getting converted into integers in the python script

### =======  Specify model & pooling ======= 
# - model is always 'LogisticRegression', as XGBoost is not differentiable
# - pool_method as also always 'mean_pooling', as attention_pooling was not implemented


### =======  Specify internal_batch_size ======= 
# - Set it to '1' if inputs are longer (data_inclusion_type = 'all days' or 'baseline_vars'), as internal_batch_size increases GPU memory requirements
# - Set it to '4' on a GPU with 80GBs of RAM for data_inclusion_type='baseline_last_day'

### =======  Specify ther_arm_duration ======= 
# - Defines which cohort of patients will be used 
# - ther_arm_duration: '4-month' or '6-month'

######## ======================================== RECOMMENDATIONS ON TWO PREDICTION LABELS WITH A100/H100 80GB GPUS ========================================
## FOR EXACT RECOMMENDATIONS PLEASE CHECK run_s9_6_integrated_gradients.slurm
# - As the distibutions of participants across therapy durations (4-month vs 6-month) is around 3:2, adjust the recommended time requested for the slurm script #   by this amount --> 4-month: 3/5 , 6-month: 2/5 of recommended time listed in  run_s9_6_integrated_gradients.slurm. 
#.  ==> Reminder: requested time is controlled by the line "#SBATCH --time=...." ! 




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

## A specific python within the desired environment (here: scarches)is executing the python file ==> this needs to be adjusted if environment or HPC changes
/data/gpfs/projects/punim2121/anaconda3/envs/scarches/bin/python  s9_12_LLM_survival_integrated_gradients.py \
                                                                --dataset_name $dataset_name\
                                                                --data_inclusion_type $data_inclusion_type \
                                                                --model $model \
                                                                --period_end_day $period_end_day \
                                                                --internal_batch_size $internal_batch_size \
                                                                --pool_method $pool_method \
                                                                --ther_arm_duration $ther_arm_duration

                    #--cpu_cores 0-3
