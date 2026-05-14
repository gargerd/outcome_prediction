import pandas as pd
import pickle
import json
from matplotlib import pyplot as plt
import torch
import numpy as np
import os
import itertools
import seaborn as sns
import warnings
from sklearn.metrics import RocCurveDisplay
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300

from s9_3_baseline_ML_models_functions import *




### EXTRACT PARAMETERS FOR PARAMETER SEARCH FROM ARGPARSE
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", help="One of the dataset names from the keys of parameters_for_analysis dictionar (see below)")
parser.add_argument("--training_data_type", help="One of the 2 data inclusion types: ['all_days_in_period','last_therapy_day']]")
parser.add_argument("--model", help="One of the 2 models: ['XGBoost','LogisticRegression']")
parser.add_argument("--period_end_day", help="One of the 7 periods: ['baseline',31,62,93,125,160,'all']")
#parser.add_argument("--overwrite_existing_params", help="If set to True, overwrites the existing dictionaries with the parameter search results")
#parser.add_argument("--cpu_cores", help="List of integers, setting which CPU cores to be used. i.e. for the first 4 CPU cores: 0-3")


args = parser.parse_args()
dataset_name_=[args.dataset_name]
training_data_type_ = [args.training_data_type]
model_names_ = [args.model]
period_end_day_=[args.period_end_day if args.period_end_day in ['baseline','all'] else int(args.period_end_day)]
#overwrite_existing_params=args.overwrite_existing_params
#print(overwrite_existing_params)



parameters_for_analysis={'tb21_22_2984_pats_22_vars_result_at_end_of_treatment':{
                            'fn':'tb21_22_2984_pats_22_vars_result_at_end_of_treatment',
                            'result_cat':'RESULT_AT_END_OF_TREATMENT'},
            
                        'tb21_22_2984_pats_22_vars_relapse':{
                            'fn':'tb21_22_2984_pats_22_vars_result_at_end_of_treatment',
                            'result_cat':'RELAPSE'}, 

                         'tb21_22_2984_pats_22_vars_relapse_ext_pats':{
                            'fn':'tb21_22_2984_pats_22_vars_result_at_end_of_treatment',
                            'result_cat':'RELAPSE'}, 

                         'tb21_22_2984_pats_22_vars_relapse_without_dr_reg_ext_pats':{
                            'fn':'tb21_22_2984_pats_22_vars_result_at_end_of_treatment',
                            'pat_ids_fn':'tb21_22_2984_pats_22_vars_relapse_ext_pats',
                            'result_cat':'RELAPSE'},

                         
                        'tb20_21_22_2905_pats_8_vars_relapse_ext_pats':{
                              'result_cat':'RELAPSE',
                            'fn':'tb20_21_22_2905_pats_8_vars_relapse',
                           'include_rifaquin':True},

                         'tb20_21_22_2905_pats_8_vars_relapse_ext_pats_no_rifaquin':{
                              'result_cat':'RELAPSE',
                            'fn':'tb20_21_22_2905_pats_8_vars_relapse',
                             #'pat_ids_fn':'tb21_22_2984_pats_22_vars_relapse_ext_pats',
                           'include_rifaquin':False},


                        'tb21_22_2984_pats_22_vars_result_at_end_of_treatment_dr_reg_per_arm':{
                            'fn':'tb21_22_2984_pats_22_vars_result_at_end_of_treatment',
                             'pat_ids_fn':'tb21_22_2984_pats_22_vars_result_at_end_of_treatment',
                            'result_cat':'RESULT_AT_END_OF_TREATMENT'},
            
                        'tb21_22_2984_pats_22_vars_relapse_dr_reg_per_arm':{
                            'fn':'tb21_22_2984_pats_22_vars_result_at_end_of_treatment',
                            'pat_ids_fn':'tb21_22_2984_pats_22_vars_relapse',
                            'result_cat':'RELAPSE'}, 

                         
                         'tb21_22_2984_pats_22_vars_result_at_end_of_treatment_with_arm':{
                            'fn':'tb21_22_2984_pats_22_vars_result_at_end_of_treatment',
                             'pat_ids_fn':'tb21_22_2984_pats_22_vars_result_at_end_of_treatment',
                            'result_cat':'RESULT_AT_END_OF_TREATMENT'},
            
                        'tb21_22_2984_pats_22_vars_relapse_with_arm':{
                            'fn':'tb21_22_2984_pats_22_vars_result_at_end_of_treatment',
                            'pat_ids_fn':'tb21_22_2984_pats_22_vars_relapse',
                            'result_cat':'RELAPSE'}, 

                        'tb21_22_2984_pats_22_vars_relapse_without_dr_reg':{
                            'fn':'tb21_22_2984_pats_22_vars_result_at_end_of_treatment',
                            'pat_ids_fn':'tb21_22_2984_pats_22_vars_relapse',
                            'result_cat':'RELAPSE'},
                         
                          'tb21_22_2984_pats_22_vars_relapse_basic_vars':{
                            'fn':'tb21_22_2984_pats_22_vars_result_at_end_of_treatment',
                            'pat_ids_fn':'tb21_22_2984_pats_22_vars_relapse',
                            'result_cat':'RELAPSE'},
                         
                        'tb21_22_2263_pats_24_vars_relapse':{
                            'fn':'tb21_22_2263_pats_24_vars_relapse',
                            'result_cat':'RELAPSE'},

                         'tb21_22_2984_pats_22_vars_result_at_end_of_treatment_with_adherence':{
                            'fn':'tb21_22_2984_pats_22_vars_result_at_end_of_treatment',
                            'result_cat':'RESULT_AT_END_OF_TREATMENT'},
            
                        'tb21_22_2984_pats_22_vars_relapse_with_adherence':{
                            'fn':'tb21_22_2984_pats_22_vars_result_at_end_of_treatment',
                            'result_cat':'RELAPSE'}, 
  

                         'tb21_22_2984_pats_22_vars_result_at_end_of_treatment_mb_only':{
                            'fn':'tb21_22_2984_pats_22_vars_result_at_end_of_treatment',
                            'result_cat':'RESULT_AT_END_OF_TREATMENT'},

                         'tb21_22_2984_pats_22_vars_result_at_end_of_treatment_without_mb':{
                            'fn':'tb21_22_2984_pats_22_vars_result_at_end_of_treatment',
                            'result_cat':'RESULT_AT_END_OF_TREATMENT'},

                         'tb21_22_2984_pats_22_vars_relapse_mb_only':{
                            'fn':'tb21_22_2984_pats_22_vars_result_at_end_of_treatment',
                            'result_cat':'RELAPSE'},

                         'tb21_22_2984_pats_22_vars_relapse_without_mb':{
                            'fn':'tb21_22_2984_pats_22_vars_result_at_end_of_treatment',
                            'result_cat':'RELAPSE'},

                         'tb21_1405_pats_40_vars_result_at_end_of_treatment':{
                            'fn':'tb21_1405_pats_40_vars_result_at_end_of_treatment',
                            'result_cat':'RESULT_AT_END_OF_TREATMENT'},
                         
                         'tb21_1405_pats_40_vars_relapse':{
                            'fn':'tb21_1405_pats_40_vars_result_at_end_of_treatment',
                            'result_cat':'RELAPSE'},

                         'tb22_1499_pats_31_vars_result_at_end_of_treatment':{
                             'fn':'tb22_1499_pats_31_vars_result_at_end_of_treatment',
                            'result_cat':'RESULT_AT_END_OF_TREATMENT'},
                         
                         'tb22_1499_pats_31_vars_relapse':{
                             'fn':'tb22_1499_pats_31_vars_result_at_end_of_treatment',
                            'result_cat':'RELAPSE'},


                         'tb21_22_2984_pats_22_vars_raw_pred_prob_norm_loss':{
                            'fn':'tb21_22_2984_pats_22_vars_result_at_end_of_treatment',
                            'result_cat':'raw_pred_prob_norm'},
                         
                        'tb21_22_2984_pats_22_vars_llm_pred_prob_norm_loss':{
                            'fn':'tb21_22_2984_pats_22_vars_result_at_end_of_treatment',
                            'result_cat':'llm_pred_prob_norm'},
                         
                        }

###=========================================================================================
# 1 . Define training parameters

from sklearn.model_selection import train_test_split
import warnings
from tqdm import tqdm


outcome_df=pd.read_csv('../data/tb_1018_20_21_22_30_outcome.csv.gz',index_col=0)
outcome_df=outcome_df.set_index('USUBJID',drop=True)
outcome_df=outcome_df.rename(columns={'UNFAVOURABLE_OUTCOME_CATEGORY_AT_18_MONTHS':'UNFAVOUR_CAT_AT_18_MONTHS'})


model_names=['XGBoost','GradientBoost','RandomForest','LogisticRegression','SVC']#,'KNN']
model_names=['XGBoost','GradientBoost','RandomForest','LogisticRegression'][:]
model_names=['XGBoost','LogisticRegression'][:]

training_data_types=['all_days_in_period','last_therapy_day']
columns_to_drop=['ARM','STUDYID','DAY','index']

temp_cols_to_drop=['ae','mh','cm','ce'][:-1]

## Return a dictionary containing the race of the patients
race_dict=return_race_dict()

## Set up prediction labels
id2label={0: "FAVOURABLE", 1: "UNFAVOURABLE"}
label2id={"FAVOURABLE": 0, "UNFAVOURABLE": 1}

#={'UNFAVOUR_CAT_AT_18_MONTHS':'UNFAVOURABLE_OUTCOME_CATEGORY_AT_18_MONTHS'}

## Drop patients who have their last data at an earlier timepoint than threshold
therapy_day_thr=80

period_end_days=['baseline',31,62,93,125,160,'all']

## Define training parameters
train_params={'num_cv_repeats':25,
                'k_folds':5,             
              'weight_by_label_freq':True,
               'label_weights':[1,1],## [index_0: weight for label 0 (negative),index_1: weight for label 1 (positive)], only
                                     ## only considered if weight_by_label_freq=False !
              'random_state':42,
              'test_size_ratio':0.2,
              'label2id':label2id}


## Load dataframe containing the last day of drug regimen for each patient
last_initial_therapy_day_df=pd.read_csv('../data/out_last_initial_therapy_day_list_1018_20_21_22_30.csv.gz',index_col=0)
last_initial_therapy_day_df=last_initial_therapy_day_df.set_index('USUBJID')

## Laod pats with relapse df
#pats_with_relapse_df=extract_21_22_relapse_pats()
pats_with_relapse_df=extract_21_22_relapse_pats(include_rifaquin=True,
                                               extended_pats=True)





###=========================================================================================
# 2. Run parameter search for models
import time    
start=time.time()
import warnings
warnings.filterwarnings("ignore")


    
param_search_dict={'RandomForest':{'n_estimators':[300,500,700],
                                   'max_features':['sqrt'],
                                   'max_depth':[3,5,7,9]},
                   
                  'GradientBoost':{'n_estimators':[300,500,700],
                                   'max_features':['sqrt'],
                                   'learning_rate':[0.1,0.3,0.5,0.8]},
                  
                  'XGBoost':{'n_estimators':[300,500,700],
                             'max_depth':[3,5,7,9],
                             'eta':[0.1,0.3,0.5,0.8],
                             #'subsample':[1.0,0.9,0.8,0.7],
                             #'tree_method':['exact'],
                             #"device": ["cpu"],
                              #'n_jobs':[1]
                            },
                   
                  'LogisticRegression':{#'l1_ratio':[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
                                        #'l1_ratio':[1]
                                        'l1_ratio':[1],
                                         'C': [0.001, 0.01, 0.1, 1, 10],
                                          },
                   
                  'SVC':{'C':[1e-3,1e-2,1e-1,1e0,1e1,1e2],
                        'kernel':['rbf','poly']},
                   
                  'KNN':{'n_neighbors':[2,5,10,25,50,100]},
                  }   




### loop over ML models, train them & training results in a dictionary
for data_param_key in dataset_name_:
    print('===========\n',data_param_key,'\n')

    ## Create dictionary to save the final patient IDS for analysis for each period
    final_pat_ids_for_analysis={}
    
    #outcome_label=data_param_key.split('vars_')[-1].upper()
    outcome_label = parameters_for_analysis[data_param_key]['result_cat']

     ## LOAD FINAL PATIENT IDS FOR ANALYSIS, SAVED DURING PREPROCESSING OF THE BASELINE MODELS IN NOTEBOOK S9_3
    if 'pat_ids_fn' in parameters_for_analysis[data_param_key].keys():
        fn=f"../data/{parameters_for_analysis[data_param_key]['pat_ids_fn']}_final_pat_ids_for_analysis.pickle"
    else:  
        fn=f'../data/{data_param_key}_final_pat_ids_for_analysis.pickle'
    with open(fn, 'rb') as handle:
        final_pat_ids_for_analysis=pickle.load(handle)

    ## Load preprocessed-imputed data, and modify the variables (add or drop) depending on the prediction setup, which is contained at the 
    #. end of the "data_param_key" variable
    X,race_colnames = load_and_modify_preprocessed_data(data_param_key)

    ## Return dataframe with the outcome label
    pat_ids,y,target_df,outcome_label = return_predict_label_dataframe(parameters_for_analysis,data_param_key,X,
                                                                  outcome_df,outcome_label,model_names)
        
    #print('pat_ids',len(pat_ids))
    
    df_=target_df.reset_index()#
    df_['STUDYID']=df_['USUBJID'].str.split('/',expand=True)[0].values
    
    ## Subset initial therapy last day dataframe to all patient considered in analysis
    #init_ther_df=last_initial_therapy_day_df.loc[pat_ids,:]
    last_init_ther_days = extract_last_init_therapy_day_from_drug_regimen(pat_ids)


    
    for period_end_day in period_end_day_[:]:  

        
        period_num = period_end_days.index(period_end_day)

        ## SUBSET TO PATIENTS WHO WERE TAKING DRUGS DURING THE PERIOD
        #  ==> keep only patients who took TB drugs in period (not only placebo)
        #  ==> for EOT outcome prediction: 
        #      patients in 4-month arms considered in Baseline-Month 3
        #.     patients in 6-month arms considered in Baseline-Month 5
        #pat_ids_ = subset_pats_with_therapy_in_period(period_num,period_end_days)
        pat_ids_ = subset_pats_with_therapy_in_period(period_num,period_end_days,last_init_ther_days,
                                       pats_with_relapse_df,period_end_day,X,outcome_label,data_param_key)
        print(len(pat_ids_))

        if len(pat_ids_)==0:
            print(f'No patients considered for {data_param_key} - period:{period_end_day}')
            continue
        
    

        df_=target_df.reset_index()#
        df_['STUDYID']=df_['USUBJID'].str.split('/',expand=True)[0].values
        #print(pd.crosstab(df_.loc[df_['USUBJID'].isin(pat_ids_),'STUDYID'],df_.loc[df_['USUBJID'].isin(pat_ids_),outcome_label]))
        
        ## Drop patients, whose therapy ended before the period_end_day & keep clinical data up until last day of period
        if isinstance(period_end_day,int):
            
            ## Select last visit in period cutoff based on threshold before cutoff & after cutoff
            X_subset = select_visits_with_dual_thresholds(
                                                df=X[X['USUBJID'].isin(pat_ids_)],
                                                time_col='DAY',
                                                patient_col='USUBJID',
                                                cutoff_day=period_end_day,
                                                before_threshold=20,
                                                after_threshold=10)

        if period_end_day=='baseline': 

            ## Use first month of data ==> used for imputation of some baseline variables where neeeded
            X_subset=X[(X['USUBJID'].isin(pat_ids_)) & \
                       ~(X['STUDYID'].isin(['TB-1018']))&\
                       (X['DAY']<=31)].copy()

        if period_end_day=='all': 
            
            X_subset=X[(X['USUBJID'].isin(pat_ids_))&\
                      ~(X['STUDYID'].isin(['TB-1018']))].copy()

        ## Drop all columns that contain only zeroes
        X_subset=X_subset.loc[:, (X_subset!= 0).any(axis=0)]

        ## Add cumulative adverse events columns 
        #X_subset,ae_cumul_colnames = calucate_cumul_adverse_clinical_events(X_subset,period_end_day)
       

        ## Define baseline columns to keep 
        temp_col_threshold=0.3 # 0.3 
        temporal_data_names=['re','ae','cm','su','mh'][:] 
        temp_cols_to_keep=['dr_reg_study_drugs_cumul','vs_Height_STD_NUM_RESULT',
                           'vs_BMI_STD_NUM_RESULT','mb_LJ-culture_CULTURE_STATUS'][:] \
                             + race_colnames #\
                             #+ ae_cumul_colnames #\
                            #+ arm_cumul_colnames \
        
        for temp_data_name in temporal_data_names:
            cols_to_keep=select_temporal_cols_with_suff_pat_data(temp_data_name,X_subset,temp_col_threshold)
            temp_cols_to_keep.extend(cols_to_keep)

        
        for n, model_name in enumerate(tqdm(model_names_[:], total=len(model_names_[:]),position=0, leave=True,unit="model")):
            print('\n=====================')
            print(model_name)

            training_results={}
            training_results['train_params']=train_params
            training_results['param_search_results']={}

            for training_data_type in training_data_type_:



                for cv_repeat_num in range(train_params['num_cv_repeats']):
                    rand_state=train_params['random_state'] + cv_repeat_num

                    print(f'++++++++++++++++++ \n{data_param_key} - Model: {model_name} - Period: {period_end_day} days  - data for training: {training_data_type} - CV:{cv_repeat_num+1}\n+++++++++++++++++')

                    if len(columns_to_drop)>0:
                        columns_to_drop_=columns_to_drop + X_subset.columns[X_subset.columns.str.startswith(tuple(temp_cols_to_drop))].tolist()

                        ## If colunm name is in the temp_cols_to_keep list, don't drop it
                        columns_to_drop_=[coln for coln in columns_to_drop_ if coln not in temp_cols_to_keep]


                    ## Forward fill cumulative drug regimen columns,with last cumulative dose for days, where drug was not taken anymore
                    #X_subset_=ffill_dr_reg_cumul_cols(X_subset)


                    ## 1. Drop patients with no drug regimen data
                    ## 2. Drop rows, where the variables columns selcted for analysis contains NaNs
                    
                    if period_end_day!='baseline':

                        ## 1. 
                        dr_reg_cols=X_subset.columns[X_subset.columns.str.contains('dr_reg')].tolist()
                        pats_wo_drug_reg = X_subset.loc[X_subset[dr_reg_cols].isna().any(axis=1),'USUBJID'].unique()
                        X_subset_=X_subset.loc[~X_subset['USUBJID'].isin(pats_wo_drug_reg),:].copy()

                        ## 2. 
                        cols_for_anal=X_subset_.drop(columns=columns_to_drop_).columns.tolist()
                        #X_subset_ = X_subset[['DAY']+cols_for_anal].sort_values(by=['DAY']).groupby('USUBJID',as_index=False).apply(lambda x: x.loc[x.index[-1],:]).dropna(how='any',axis=1)
                        #X_subset_ = X_subset.dropna(subset=cols_for_anal,how='any',axis=0)
                        X_subset_=X_subset_[['DAY']+cols_for_anal].copy()
                        
                        ## Drop all columns that contain only zeroes and refreash the columns_to_drop_ list with columns that are still there in X_susbet_
                        X_subset_=X_subset_.loc[:, (X_subset_!= 0).any(axis=0)]
                        columns_to_drop_= list(set(columns_to_drop_)&set(X_subset_.columns))

                    ## If baseline, don't drop these rows, as they are being used to impute some variables at baseline
                    if period_end_day=='baseline':
                        X_subset_=X_subset.copy()
                        X_subset_ = X_subset_.loc[:,~X_subset_.columns.str.startswith('ARM_')]

                    #final_pat_ids_for_analysis[period_end_day]=X_subset_['USUBJID'].unique().tolist()
                    
                    
                
                
                    ## Split into train-test data
                    X_train,X_test,y_train,y_test,_,_ = create_std_training_testing_data(X_subset_,
                                                                                         y,
                                                                                         pat_ids_,
                                                                                         train_params['test_size_ratio'],
                                                                                         rand_state,training_data_type,
                                                                                         columns_to_drop_,
                                                                                         period_end_day,
                                                                                         outcome_label,
                                                                                         cv_repeat_num=cv_repeat_num,
                                                                                         final_pat_ids_for_analysis=final_pat_ids_for_analysis)
        
                    final_pat_ids_for_anal=X_train.index.unique().tolist() + X_test.index.unique().tolist()
                    print(pd.crosstab(df_.loc[df_['USUBJID'].isin(final_pat_ids_for_anal),'STUDYID'],\
                                      df_.loc[df_['USUBJID'].isin(final_pat_ids_for_anal),outcome_label]))
                    print('pat_ids_',len(pat_ids_),'final_pat_ids_for_anal',len(final_pat_ids_for_anal))
                    #print(.shape[0],X_test.index.unique().shape[0])
                    #print(X_train.columns)
                    X_train, X_test=scale_by_training_data(X_train, X_test)


                    ## Save train-test ids to be consistent with CV-splits in the LLM models later
                    final_pat_ids_for_analysis[period_end_day][cv_repeat_num]={}
                    final_pat_ids_for_analysis[period_end_day][cv_repeat_num]['X_train_ids']=X_train.index.tolist()
                    final_pat_ids_for_analysis[period_end_day][cv_repeat_num]['X_test_ids']=X_test.index.tolist()
                    
                    #df_=y.reset_index()#
                    #print(pd.crosstab(df_['STUDYID'],df_[outcome_label]))    
                    
                    #print(X_train.columns)
                    
                    print(f'Running training with {training_data_type} model: {X_train.shape[1]} vars')
                    print(f"Num of CV-repeat:{cv_repeat_num+1}")
                    #training_results['param_search_results'][f'cv_rep_{cv_repeat_num}']={}
                    
                
                    ## RUN PARAMETER SEARCH
                    calibrate_model=False
                    param_search_results = run_parameter_search(model_name,
                                                                X_train,
                                                                y_train,
                                                                train_params['k_folds'],
                                                                train_params['random_state'],
                                                                outcome_label,
                                                                param_search_dict,
                                                                train_params['weight_by_label_freq'],
                                                                train_params,
                                                                calibrate_model)
                    
                    

                    training_results['param_search_results'][f'cv_rep_{cv_repeat_num}']=param_search_results
                    

                
                
                
                ## Save training results
                #fn=f'../data/{data_param_key}_{model_name}_{training_data_type}_training_results.pickle'
                fn=f'../data/{data_param_key}_{model_name}_{period_end_day}_days_{training_data_type}_param_search_results.pickle'
                with open(fn, 'wb') as handle:
                    pickle.dump(training_results, handle)

        
        #if training_data_type=='last_therapy_day':              
            ## SAVE DICTIONARY OF FINAL PATIENT IDS                    
        #    fn=f'../data/{data_param_key}_final_pat_ids_for_analysis.pickle'
        #    with open(fn, 'wb') as handle:
        #        pickle.dump(final_pat_ids_for_analysis, handle)
        
        
loop_time=time.time()
print('Training duration:')
print_elapsed_time(start,loop_time) 
                
                    
                    

