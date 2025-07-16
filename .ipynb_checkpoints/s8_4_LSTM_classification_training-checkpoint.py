import pandas as pd
import warnings
# Ignoring FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)
#warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)
import numpy as np
import os
import time
import pickle
from matplotlib import pyplot as plt
import seaborn as sns
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score,f1_score,mean_squared_error,roc_auc_score
from sklearn.model_selection import GridSearchCV,KFold,train_test_split

from itertools import chain
import math

from sklearn.preprocessing import StandardScaler,MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader,Dataset
from torch.nn.utils.rnn import pad_sequence,pack_padded_sequence,pad_packed_sequence
from torch.optim.lr_scheduler import CosineAnnealingLR, CyclicLR
from torch.optim.swa_utils import AveragedModel
#from pytorch_model_summary import summary

from s8_4_LSTM_classification_functions import *


### EXTRACT PARAMETERS FOR PARAMETER SEARCH FROM ARGPARSE
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", help="One of the dataset names from the keys of parameters_for_analysis dictionar (see below)")
#parser.add_argument("--data_inclusion_type", help="One of the 3 data inclusion types: ['baseline_last_day','baseline_vars','all_days']")
parser.add_argument("--model_complexity", help="One of the 3 model complexities:['day_only','regimen_only','full']")
parser.add_argument("--period_end_day", help="One of the 7 periods: ['baseline',31,62,93,125,160,'all']")
#parser.add_argument("--overwrite_existing_params", help="If set to True, overwrites the existing dictionaries with the parameter search results")
#parser.add_argument("--cpu_cores", help="List of integers, setting which CPU cores to be used. i.e. for the first 4 CPU cores: 0-3")


args = parser.parse_args()
dataset_name_=[args.dataset_name]
#data_inclusion_type_ = [args.data_inclusion_type]
model_complexity_ = [args.model_complexity]
period_end_day_=[args.period_end_day if args.period_end_day in ['baseline','all'] else int(args.period_end_day)]
#overwrite_existing_params=args.overwrite_existing_params
#print(overwrite_existing_params)





1. ## Determine which preprocessed dataset to load

parameters_for_analysis={'tb21_22_2984_pats_22_vars_result_at_end_of_treatment':{
                            'fn':'tb21_22_2984_pats_22_vars_result_at_end_of_treatment',
                              'training_days':120,
                            'result_cat':'RESULT_AT_END_OF_TREATMENT'},
            
                        'tb21_22_2984_pats_22_vars_relapse':{
                            'fn':'tb21_22_2984_pats_22_vars_result_at_end_of_treatment',
                              'training_days':120,
                            'result_cat':'RELAPSE'}

                         }

## Return a dictionary containing the race of the patients
race_dict=return_race_dict()

## Drop patients who have their last data at an earlier timepoint than threshold
therapy_day_thr=80

period_end_days=['baseline',31,62,93,125,160,'all']

## Set up prediction labels
id2label={0: "FAVOURABLE", 1: "UNFAVOURABLE"}
label2id={"FAVOURABLE": 0, "UNFAVOURABLE": 1}


import time

outcome_df=pd.read_csv('../data/tb_1018_20_21_22_30_outcome.csv.gz',index_col=0)
outcome_df=outcome_df.set_index('USUBJID',drop=True)
outcome_df=outcome_df.rename(columns={'UNFAVOURABLE_OUTCOME_CATEGORY_AT_18_MONTHS':'UNFAVOUR_CAT_AT_18_MONTHS'})

## List of temporal dataframe names to drop from loaded dataframe (as they are highly sparse, perhaps only add noise?)
temporal_data_names=['ae_','cmind_','cmdos_','cmday_','mh']

#columns_to_drop=['USUBJID','ARM','STUDYID','DAY','index']
columns_to_drop=['ARM','STUDYID','DAY','index'][:-1]
temp_cols_to_drop=['ae','ce','mh','cm']

therapy_day_thr=80

## Load pat IDS with information which study phase they belong to
#pat_id_df=pd.read_csv('../data/patients_in_analysis.csv',index_col=0)

data_split_random_state=42
training_method='training_on_first_x_days'
training_methods=['training_on_whole_data','training_on_first_x_days']

y_as_log=False
y_min_max=False




####================================================================================================================================
# 3.1 Parameter search & training parameters for LSTM

## List of prediction models
model_names=['LSTM']
scoring_methods=['f1','precision','recall']
scoring_methods=['roc_auc_score']#,'precision','recall']


## LOAD DATAFRAME CONTATINING TARGET OUTCOMES
outcome_df=pd.read_csv('../data/tb_1018_20_21_22_30_outcome.csv.gz',index_col=0)
outcome_df=outcome_df.set_index('USUBJID',drop=True)
outcome_df=outcome_df.rename(columns={'UNFAVOURABLE_OUTCOME_CATEGORY_AT_18_MONTHS':'UNFAVOUR_CAT_AT_18_MONTHS'})

therapy_day_thr=80
period_end_days=['baseline',31,62,93,125,160,'all']


## Set up prediction labels
id2label={0: "FAVOURABLE", 1: "UNFAVOURABLE"}
label2id={"FAVOURABLE": 0, "UNFAVOURABLE": 1}


scoring_methods_dict={'roc_auc_score':sklearn.metrics.roc_auc_score}


## Categorical mapping of MGIT values. Based on the MGIT values assign categorical value to predict
#  i.e. MGIT <43 days : positive (==1), days >=43: negative (==0)
categorical_map_dict={'labels':[1,0],
                      'labels_str':['UNFAVOURABLE','FAVOURABLE'],
                      'weights':[1,1]}


## Set training method and number of days to train on if training method is 'training_on_first_x_days'
training_methods=['training_on_whole_data']#,'training_on_first_x_days']
random_state=10
data_split_random_state=42


## Boolean if we want to update exsiting result_dict -> if False, the training will be run on only not-yet existing 
#  +ds_name+'_preproc_data.csv' files
update_existing_dicts=True

## List of model complexities: complexity here means which features are used as input for the model
# - LR: logistic regression with STUDY DAY as input ONLY  
# - BASE: multiple models with DRUG REGIMEN ONLY ==> NO Study day information
# - FULL: multiple models with ALL PATIENT DATA + DRUG REGIMEN  ==> NO Study day information
model_complexities=['day_only','regimen_only','full']


#columns_to_drop=['USUBJID','STUDYID','DAY','ARM','index'] #'ARM'
cat_vars_not_to_hot_encode=['STUDYID','ARM','USUBJID'][-1:]


####================================================================================================================================

###### === CREATE HYPERPARAMETER SEARCH GRID 

from itertools import product
import copy

# Base LSTM parameters (unchanging defaults)
lstm_parameters = {
    'batch_size': 128,
    'hidden_size': 128,
    'fc_hidden_dims': [64, 32, 8],
    'num_of_lstm_layers': 1,
    'output_size': len(categorical_map_dict['labels']),
    'dropout_prob': 0.3,
    'bidirectional': True,
    'weight_Cross_Entropy_by_label_freq': True,
    'weight_celoss': 0,
    'weight_mse_loss': 1,
    'criterion': 'CrossEntropyLoss',

    ## Stochastic weight averaging params
    ## https://pytorch.org/blog/pytorch-1.6-now-includes-stochastic-weight-averaging/
    
    ## Pretraining with AnnealingCosine LR scheduler => after pretraining CyclicLR, where every last
    #  swa_update_in_epoch_ratio * number_of_batches element in an epoch will update the AverageModel
    #  i.e. fastSWA: https://sh-tsang.medium.com/review-there-are-many-consistent-explanations-of-unlabeled-data-why-you-should-average-edcf5b3bfd7d
    
    'pretrain_max_lr': 1e-4,
    'pretrain_min_lr': 1e-4,
    'pretrain_weight_decay': 1e-4,
    'pretrain_epochs': 3,
    'swa_epochs': 0,
    'swa_max_lr': 1e-2,
    'swa_min_lr': 1e-4,
    'swa_update_in_cycle_ratio': 0.2
}

# Define the grid of parameters to search
param_grid = {
    'pretrain_epochs':[100,200],
    'num_of_lstm_layers': [1, 2, 4][:],
    'hidden_size': [64, 128][:]}

# Generate all combinations of the specified parameters
keys, values = zip(*param_grid.items())
combinations = [dict(zip(keys, v)) for v in product(*values)]


## Based on the LSTM hidden size, dynamically compute the hidden dimensions of the ultimate FC layers
def create_fc_hidden_dims(hs):
    fc_hidden_dims= [hs ,hs // 2, hs // 8]
    return fc_hidden_dims

# Create a list of parameter dictionaries for each combo
parameter_sets = {}
for n,combo in enumerate(combinations):
    new_params = copy.deepcopy(lstm_parameters)  # Avoid modifying the original
    new_params.update(combo)
    
    # Dynamically compute fc_hidden_dims
    new_params['fc_hidden_dims'] = create_fc_hidden_dims(new_params['hidden_size'])
    parameter_sets[n] = new_params

## Create dataframe containing hyperparameters
grid_search_df=pd.DataFrame.from_dict(parameter_sets).T
grid_search_df.index.name='param_combnum'






####================================================================================================================================

# 3.2 Run LSTM parameter search

import warnings
warnings.filterwarnings("ignore")

pats_with_relapse_df=extract_21_22_relapse_pats()


import warnings
warnings.filterwarnings("ignore")



## Choose CV features: number of folds and repeats for CV in the training + scoring method
#### SET THIS TRUE FOR CV!!!! #########
cross_validation=False
#cross_validation=True

## If CV==False, set the numbers to 1 (they determine the number of rows and cols when plotting the results later)
if cross_validation==False:
    num_of_cv_repeats=25
    k_folds=None
    
    
if cross_validation==True:
    ## Choose CV features: number of folds and repeats for CV in the training + scoring method
    num_of_cv_repeats=25
    k_folds=2


start=time.time()

for model_complex in model_complexity_:
    print('++++++++++++++++++')
    print('Model type:',model_complex)
    
    for data_param_key in dataset_name_:

        outcome_label=data_param_key.split('vars_')[-1].upper()
        
        ## Select the number of training days for the training method 'training_on_first_x_days'
        parameter_dict=parameters_for_analysis[data_param_key]
        training_days=parameter_dict['training_days']
        print('dataset name: ',data_param_key)

        ## Load the preprocessed + imputed dataset
        data_dir='../data'
         ## Define X and y dataframes for training
        fn='../data/'+parameters_for_analysis[data_param_key]['fn']+'_preproc_data_with_imp.csv.gz'
        data=pd.read_csv(fn,index_col=0,low_memory=False)

        ## If not RELAPSE shpuld be predicted, subset the patients according to the availbility of the outcome results
        if parameters_for_analysis[data_param_key]['result_cat']!='RELAPSE':
            
            ## Extract patients who have their last therapy day before therapy_day_thr ==> these patient probably dropped out
            last_day_per_pat_df=data.sort_values(by=['DAY']).groupby('USUBJID').apply(lambda x: x.loc[x.index[-1],:])
            pat_ids=last_day_per_pat_df[last_day_per_pat_df['DAY']>therapy_day_thr]['USUBJID'].tolist()
            #pat_ids=X['USUBJID'].unique().tolist()
            
            ## Subset outcome dataframe to patient considered
            target_df=outcome_df.loc[pat_ids,outcome_label]
            y=target_df.loc[pat_ids].replace(label2id)
            
        if parameters_for_analysis[data_param_key]['result_cat']=='RELAPSE':
            outcome_label='RELAPSE'
            pats_with_relapse_df=extract_21_22_relapse_pats()
    
            pats_with_relapse_df = pats_with_relapse_df.loc[list(set(data['USUBJID'].unique())&set(pats_with_relapse_df.index))]
    
            ## Create new prediction labels (or even multilabels) in the "RELAPSE" column based on the relapse day intervals defined in "bins" 
            if 'bins' in parameters_for_analysis[data_param_key].keys():
                pats_with_relapse_df = cut_relapse_days_to_interval_categories(pats_with_relapse_df,data_param_key)
            
            pat_ids=pats_with_relapse_df.index.tolist()
            target_df=pats_with_relapse_df[[outcome_label]]
            y=pats_with_relapse_df[[outcome_label]]

        ## Subset initial therapy last day dataframe to all patient considered in analysis
        #init_ther_df=last_initial_therapy_day_df.loc[pat_ids,:]
        last_init_ther_days = extract_last_init_therapy_day_from_drug_regimen(pat_ids)

        
        ## Subset outcome dataframe to patient considered
        outcome_label=data_param_key.split('vars_')[-1].upper()
        target_df=target_df.loc[pat_ids].replace(label2id)
            


        ## Check if result_dict already exists. If yes, load it, if not create a new one
        #fname='../data/'+data_param_key+'_LSTM_class_results.pickle'
        fname=f'../data/{data_param_key}_LSTM_class_results_{model_complex}_model.pickle'
        if os.path.exists(fname) and update_existing_dicts==False:
            with open(fname,'rb') as f:
                result_dict=pickle.load(f)
        
        ## If result_dict doesn't exist yet or we want to update the existing result_dict -> 
        #  create empty result_dict and collect the results in it
        if os.path.exists(fname)==False or update_existing_dicts==True:
            result_dict={}

        ## Check if test_data_results or cv_results already exists -> if not create them
        #for result_type in ['test_data_results','cv_results','feat_importances']:
        #    if result_type not in result_dict.keys():
        #        result_dict[result_type]={}
        

        #print('Shape of final data: ',data.shape) 
        #print()

        cross_entropy_weights_dict=dict(zip(categorical_map_dict['labels'],categorical_map_dict['weights']))


        
        for period_end_day in period_end_day_:

            print('============= Period_end_day:',period_end_day,'===============\n')     

            period_num = period_end_days.index(period_end_day)
        
            ## SUBSET TO PATIENTS WHO WERE TAKING DRUGS DURING THE PERIOD
            #pat_ids_ = subset_pats_with_therapy_in_period(period_num,period_end_days)
            pat_ids_ =  subset_pats_with_therapy_in_period(period_num,period_end_days,
                                                           period_end_day,data_param_key,
                                                           last_init_ther_days,
                                                          pats_with_relapse_df)

            data=data.loc[data['USUBJID'].isin(pat_ids_)]
            target_df_=target_df.loc[pat_ids_]



            ## Create dicts to collect test data, CV-results and importance scores for all training methods
            train_data_results={}
            test_data_results={}
            

            ## Check if test_data_results or cv_results already exists -> if not create them
            for result_type in ['test_data_results','cv_results','feat_importances']:
                if result_type not in result_dict.keys():
                    result_dict[result_type]={}

            ### TRAIN MODELS AND SAVE CV-RESULTS + TEST DATA RESULTS #####
            ## Check if data already exists for given dataset -> only run training when training hasn't been run yet
            #if training_method not in result_dict['test_data_results'].keys() \
            #    or training_method not in result_dict['cv_results'].keys():

             ## Load results of parameter search 
            fname=f'../data/{data_param_key}_LSTM_class_results_{model_complex}_model_{period_end_day}_days_param_serach_results.pickle'
            with open(fname, 'rb') as f:
                param_search_result_dict=pickle.load(f)      

            ## Extract dict containing best parameter combinations per external CV-split
            best_param_combinations = extract_best_param_comb(param_search_result_dict)

            
            ## Update previously created dicts with ML results
            train_data_results,test_data_results=train_models(data,target_df_,pat_ids_,period_end_day,cat_vars_not_to_hot_encode,
                                                              scoring_methods,outcome_label,\
                                                              training_days,random_state,data_split_random_state,\
                                                              test_data_results,train_data_results,parameter_sets,
                                                              data_param_key,cross_entropy_weights_dict,
                                                              model_complex,k_folds,
                                                              best_param_combinations,
                                                              start,
                                                              categorical_map_dict,
                                                              num_of_cv_repeats,
                                                              columns_to_drop,
                                                              verbose=False,                                                              
                                                              cross_validation=cross_validation,
                                                              get_feature_importances=False)


            ## Add results of test data and CV to the result_dict
            result_dict['test_data_results']=test_data_results
            result_dict['train_data_results']=train_data_results 
            #result_dict['lstm_parameters']=lstm_parameters
            #result_dict[data_param_key][categorical_map_name]['cv_results']=cv_results  

            ## Save results of ML on the given data
            fname=f'../data/{data_param_key}_LSTM_class_results_{model_complex}_model_{period_end_day}_days_training_results.pickle'
            with open(fname, 'wb') as f:
                pickle.dump(result_dict, f)      

print('\n\n=======================')
stop=time.time()
t=print_elapsed_time(start,stop)
print('Script duration:',t)  
print('=======================')                  



