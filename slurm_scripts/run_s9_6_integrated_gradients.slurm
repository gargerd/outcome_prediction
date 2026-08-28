#!/bin/bash
##SBATCH --array=1-84%20
#SBATCH --job-name=s_9_6_ML_LLM_integrated_gradients
#SBATCH -o slurm_IG_logs/s_9_6_ML_LLM_integrated_gradients_%A_%a.log
#SBATCH -e slurm_IG_logs/s_9_6_ML_LLM_integrated_gradients_%A_%a.log
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=16G
#SBATCH --partition=gpu-h100
##SBATCH --qos=cpu_normal
#SBATCH --gres=gpu:1
#SBATCH --time=1-03:00:00. 
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




######## ======================================== RECOMMENDATIONS ON TWO PREDICTION LABELS WITH A100/H100 80GB GPUS ========================================

###  - For 'common variables - visit at end of period' data input approach (data_inclusion_type='baseline_last_day') ==> the last-vist only approach has around 1200 tokens per patient, independent of timepoint
#.   ==> USE FOLLOWING PARAMETERS WITH A100/H100 80GB GPUS

#.  dataset_name='tb21_22_2984_pats_22_vars_result_at_end_of_treatment_without_dr_reg' - 'common variables - visit at end of period' was the best performing for EOT outcome prediction, used as final embedding-based EOT outcome models (including Sparmen-correlation comparison with raw last-visit model SHAP values, Fig. 4)
#.  dataset_name='tb21_22_2984_pats_22_vars_relapse_without_dr_reg_ext_pats' - 'common variables - visit at end of period' was NOT the best performing, used only in the Spearman correlation comparison with raw last-visit model SHAP value, Fig. 4
#.  data_inclusion_type='baseline_last_day'
#.  period_end_day='baseline 31 62 93 125 160 all'
#.  internal_batch_size='4'
#.  pool_method='mean_pooling'
#.  LINE 10: #SBATCH --gres=gpu:1
#.  LINE 11: #SBATCH --time=1-03:00:00.  ==> the last-vist only approach has around 1200 tokens per patient ==> with the parameters listed here, a bit more than 24 hourse should be enough for the script to run



### - For data input approaches with multiple visits - 'common variables / all variables - all visits in period'  (data_inclusion_type='baseline_vars' or 'all days') ==>  number of tokens increases along the timepoints, with 1200 - 2000 tokens per patient at baseline, to 5000-7000 tokens at timepoints 125 - 'all'
#.   ==> USE FOLLOWING PARAMETERS WITH A100/H100 80GB GPUS

#.  dataset_name='tb21_22_2984_pats_22_vars_relapse_without_dr_reg_ext_pats' - 'all variables - all visits in period' was the best performing, used for all the supplementary figures and supplementary tables, except for Fig. 4
#.  data_inclusion_type='all_days'
#.  internal_batch_size='1'
#.  pool_method='mean_pooling'
#.  period_end_day='baseline 31 62 93 125 160 all'
#.                  ==> as the number of tokens increase with time compared to the data_inclusion_type='baseline_last_day', for period_end_day 'baseline' I would recommend to run the script per timepoint (i.e. period_end_day='31', period_end_day='62', period_end_day='93', etc.), as one timepoint takes long time, and the GPU memory between the timepoints is not completely cleared between timepoints, leading to out-of-memory problems.  

##. ==> RECOMMENDED SETTINGS BASED ON PREVIOUS EXPERIENCE
#.  period_end_day:'baseline'; #SBATCH --time=0-08:00:00.
#.  period_end_day:'31'; #SBATCH --time=1-00:00:00.; #SBATCH --gres=gpu:1
#.  period_end_day:'62'; #SBATCH --time=1-10:00:00.; #SBATCH --gres=gpu:1
#.  period_end_day:'93'; #SBATCH --time=1-15:00:00.; #SBATCH --gres=gpu:1
#.  period_end_day:'125'; #SBATCH --time=1-20:00:00.; #SBATCH --gres=gpu:1
#.  period_end_day:'160'; #SBATCH --time=1-20:00:00.; #SBATCH --gres=gpu:2
#.  period_end_day:'all'; #SBATCH --time=1-20:00:00.; #SBATCH --gres=gpu:2



#model='LogisticRegression'
#data_inclusion_type='all_days'
#model='LogisticRegression'
#period_end_day='baseline'
#period_end_day='baseline 31 62 93 125 160 all'
internal_batch_size='1'

#dataset_name='tb21_22_2984_pats_22_vars_result_at_end_of_treatment'
#dataset_name='tb21_22_2984_pats_22_vars_result_at_end_of_treatment_without_dr_reg'
dataset_name='tb21_22_2984_pats_22_vars_relapse_without_dr_reg_ext_pats'
data_inclusion_type='baseline_last_day'
model='LogisticRegression'
period_end_day='baseline 31 62 93 125 160'
internal_batch_size='4'

pool_method='mean_pooling'


#echo ${%a}
echo ${data_inclusion_type}
echo ${period_end_day}
echo ${model}
echo ${pool_method}



## A specific python within the desired environment (here: scarches)is executing the python file ==> this needs to be adjusted if environment or HPC changes
/data/gpfs/projects/punim2121/anaconda3/envs/scarches/bin/python  s9_6_LLM_integrated_gradients_LR.py \
                                                                --dataset_name $dataset_name\
                                                                --data_inclusion_type $data_inclusion_type \
                                                                --model $model \
                                                                --period_end_day $period_end_day \
                                                                --internal_batch_size $internal_batch_size \
                                                                --pool_method $pool_method

                    #--cpu_cores 0-3
