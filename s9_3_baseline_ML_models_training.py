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


                          'tb21_22_2984_pats_22_vars_relapse_without_dr_reg':{
                            'fn':'tb21_22_2984_pats_22_vars_result_at_end_of_treatment',
                            'result_cat':'RELAPSE'},

                         

                         'tb21_22_2984_pats_22_vars_result_at_end_of_treatment_weight_norm':{
                            'fn':'tb21_22_2984_pats_22_vars_result_at_end_of_treatment',
                            'result_cat':'RESULT_AT_END_OF_TREATMENT'},
            
                        'tb21_22_2984_pats_22_vars_relapse_weight_norm':{
                            'fn':'tb21_22_2984_pats_22_vars_result_at_end_of_treatment',
                            'result_cat':'RELAPSE'}, 
                        

                         'tb21_22_2984_pats_22_vars_relapse_1_year':{
                            'fn':'tb21_22_2984_pats_22_vars_result_at_end_of_treatment',
                            'result_cat':'RELAPSE',
                             'bins':[0,365][:],
                             'labels':[1]}, 

                         'tb21_22_2984_pats_22_vars_relapse_half_years':{
                            'fn':'tb21_22_2984_pats_22_vars_result_at_end_of_treatment',
                            'result_cat':'RELAPSE',
                             'bins':[0,182,365,np.inf],
                             'labels':[1,2,3]}, 
                         
                         
                         'tb21_22_2840_pats_23_vars_result_at_end_of_treatment':{
                            'fn':'tb21_22_2840_pats_23_vars_result_at_end_of_treatment',
                            'result_cat':'RESULT_AT_END_OF_TREATMENT'},

                         'tb21_22_2840_pats_23_vars_relapse':{
                            'fn':'tb21_22_2840_pats_23_vars_result_at_end_of_treatment',
                            'result_cat':'RELAPSE'},
        

                         'tb21_22_2798_pats_24_vars_result_at_end_of_treatment':{
                            'fn':'tb21_22_2798_pats_24_vars_result_at_end_of_treatment',
                            'result_cat':'RESULT_AT_END_OF_TREATMENT'},   

                          'tb21_22_2798_pats_24_vars_relapse':{
                            'fn':'tb21_22_2798_pats_24_vars_result_at_end_of_treatment',
                            'result_cat':'RELAPSE'}, 
                         
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
pats_with_relapse_df=extract_21_22_relapse_pats()


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
                                       'l1_ratio':[1]
                                          },
                   
                  'SVC':{'C':[1e-3,1e-2,1e-1,1e0,1e1,1e2],
                        'kernel':['rbf','poly']},
                   
                  'KNN':{'n_neighbors':[2,5,10,25,50,100]},
                  }



###=========================================================================================
# 2. Run parameter search for models
import time    
start=time.time()
import warnings
warnings.filterwarnings("ignore")

#columns_to_drop=['ARM','STUDYID','DAY','index']

### loop over ML models, train them & training results in a dictionary
for data_param_key in dataset_name_:
    print('===========\n',data_param_key,'\n')
    
    #outcome_label=data_param_key.split('vars_')[-1].upper()
    outcome_label = parameters_for_analysis[data_param_key]['result_cat']
        

    ## Load preprocessed-imputed data, and modify the variables (add or drop) depending on the prediction setup, which is contained at the 
    #. end of the "data_param_key" variable
    X,race_colnames = load_and_modify_preprocessed_data(data_param_key)


    ## If not RELAPSE shpuld be predicted, subset the patients according to the availbility of the outcome results
    if parameters_for_analysis[data_param_key]['result_cat']!='RELAPSE':
        
        ## Extract patients who have their last therapy day before therapy_day_thr ==> these patient probably dropped out
        last_day_per_pat_df=X.sort_values(by=['DAY']).groupby('USUBJID').apply(lambda x: x.loc[x.index[-1],:])
        pat_ids=last_day_per_pat_df[last_day_per_pat_df['DAY']>therapy_day_thr]['USUBJID'].tolist()
        #pat_ids=X['USUBJID'].unique().tolist()
        
        ## Subset outcome dataframe to patient considered
        target_df=outcome_df.loc[pat_ids,outcome_label]
        y=target_df.loc[pat_ids].replace(label2id)
        
    if parameters_for_analysis[data_param_key]['result_cat']=='RELAPSE':
        outcome_label='RELAPSE'
        pats_with_relapse_df=extract_21_22_relapse_pats()

        pats_with_relapse_df = pats_with_relapse_df.loc[list(set(X['USUBJID'].unique())&set(pats_with_relapse_df.index))]

        ## Create new prediction labels (or even multilabels) in the "RELAPSE" column based on the relapse day intervals defined in "bins" 
        if 'bins' in parameters_for_analysis[data_param_key].keys():
            pats_with_relapse_df = cut_relapse_days_to_interval_categories(pats_with_relapse_df,data_param_key)
        
        pat_ids=pats_with_relapse_df.index.tolist()
        target_df=pats_with_relapse_df[[outcome_label]]
        y=pats_with_relapse_df[[outcome_label]]
        
       
    df_=target_df.reset_index()#
    df_['STUDYID']=df_['USUBJID'].str.split('/',expand=True)[0].values
    print(pd.crosstab(df_['STUDYID'],df_[outcome_label]))

    ## Subset initial therapy last day dataframe to all patient considered in analysis
    #init_ther_df=last_initial_therapy_day_df.loc[pat_ids,:]
    last_init_ther_days = extract_last_init_therapy_day_from_drug_regimen(pat_ids)

    
    
    for period_end_day in period_end_day_:                
        period_num = period_end_days.index(period_end_day)
        
        ## SUBSET TO PATIENTS WHO WERE TAKING DRUGS DURING THE PERIOD
        #  ==> keep only patients who took TB drugs in period (not only placebo)
        #  ==> for EOT outcome prediction: 
        #      patients in 4-month arms considered in Baseline-Month 3
        #.     patients in 6-month arms considered in Baseline-Month 5
        #pat_ids_ = subset_pats_with_therapy_in_period(period_num,period_end_days)
        pat_ids_ = subset_pats_with_therapy_in_period(period_num,period_end_days,last_init_ther_days,
                                       pats_with_relapse_df,period_end_day,X,outcome_label,data_param_key)
        
        print('pat_ids_:',len(pat_ids_))

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

        
        #X_subset,ae_cumul_colnames = calucate_cumul_adverse_clinical_events(X_subset,period_end_day)
       

        ## Define baseline columns to keep 
        temp_col_threshold=0.3 # 0.3 
        temporal_data_names=['re','ae','cm','su','mh'][:] 
        temp_cols_to_keep=['dr_reg_study_drugs_cumul','vs_Height_STD_NUM_RESULT',
                           'vs_BMI_STD_NUM_RESULT','mb_LJ-culture_CULTURE_STATUS'][:] \
                            + race_colnames #\
                            #+ ae_cumul_colnames
                            #+ arm_cumul_colnames \
                            
        
        for temp_data_name in temporal_data_names:
            cols_to_keep=select_temporal_cols_with_suff_pat_data(temp_data_name,X_subset,temp_col_threshold)
            temp_cols_to_keep.extend(cols_to_keep)
        
        for n, model_name in enumerate(tqdm(model_names_[:], desc="Processing", unit="model")):
            print('\n=====================')
            print(model_name)

            training_results={}
            training_results['train_params']=train_params
            training_results['cv_results']={}

            for training_data_type in training_data_type_:

                ## Take the mean of the best parameteres selected for each CV-split as best parameters for the final model
                metric_func=np.mean
                num_of_top_models_per_cv=1

                if model_name=='XGBoost':
                    ## LOAD RESULTS OF PARAMETER SEARCH AND EXTRACT THE PARAMETERS OF THE BEST MODEL
                    fn=f'../data/{data_param_key}_{model_name}_{period_end_day}_days_{training_data_type}_param_search_results.pickle'
                    fn=f'../data/{data_param_key}_{model_name}_{period_end_day}_days_{training_data_type}_param_search_results.pickle'
                    with open(fn, 'rb') as handle:
                        param_search_results=pickle.load(handle)

                    best_model_params_across_cvs=extract_best_model_params(param_search_results,metric_func,num_of_top_models_per_cv,
                                                                       average_models_across_splits=False)
                
                
       
                if model_name=='LogisticRegression':
                    num_of_top_models_per_cv = min(1,len(param_search_dict[model_name]['l1_ratio']))


                for cv_repeat_num in range(train_params['num_cv_repeats']):
                    rand_state=train_params['random_state'] + cv_repeat_num

                    if len(columns_to_drop)>0:
                        columns_to_drop_=columns_to_drop + X_subset.columns[X_subset.columns.str.startswith(tuple(temp_cols_to_drop))].tolist()

                        ## If colunm name is in the temp_cols_to_keep list, don't drop it
                        columns_to_drop_=[coln for coln in columns_to_drop_ if coln not in temp_cols_to_keep]

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

                    #print(pd.crosstab(df_.loc[df_['USUBJID'].isin(X_subset_['USUBJID'].unique()),'STUDYID'],df_.loc[df_['USUBJID'].isin(X_subset_['USUBJID'].unique()),outcome_label]))


                    X_train,X_test,y_train,y_test,\
                                    X_train_pat_ids,X_test_pat_ids= create_std_training_testing_data(X_subset_,
                                                                                                     y,
                                                                                                     pat_ids_,
                                                                                                     train_params['test_size_ratio'],
                                                                                                     rand_state,training_data_type,
                                                                                                     columns_to_drop_,
                                                                                                     period_end_day,
                                                                                                    outcome_label)
                    X_train, X_test=scale_by_training_data(X_train, X_test)
                    
                    final_pat_ids_for_anal=X_train.index.unique().tolist() + X_test.index.unique().tolist()
                    print(pd.crosstab(df_.loc[df_['USUBJID'].isin(final_pat_ids_for_anal),'STUDYID'],\
                                      df_.loc[df_['USUBJID'].isin(final_pat_ids_for_anal),outcome_label]))
                    print('pat_ids_',len(pat_ids_),'final_pat_ids_for_anal',len(final_pat_ids_for_anal))
                    
                    #print(X_train.columns.tolist())
                    #df_=y_train.reset_index()#
                    #print(pd.crosstab(df_['STUDYID'],df_[outcome_label]))    
                    
                    print(f'{data_param_key} - period:{period_end_day} - {training_data_type} model: {model_name} - {X_train.shape[1]} vars')
                    print(f"Num of CV-repeat:{cv_repeat_num+1}")
                    training_results['cv_results'][f'cv_rep_{cv_repeat_num}']={}

                    training_results['cv_results'][f'cv_rep_{cv_repeat_num}']['X_train_pat_ids']=X_train_pat_ids
                    training_results['cv_results'][f'cv_rep_{cv_repeat_num}']['X_test_pat_ids']=X_test_pat_ids
                    training_results['cv_results'][f'cv_rep_{cv_repeat_num}']['rand_state']=rand_state
                    #training_results['cv_results'][f'cv_rep_{cv_repeat_num}']['y_train_indices']=y_train.index.tolist()
                    #training_results['cv_results'][f'cv_rep_{cv_repeat_num}']['y_test_indices']=y_test.index.tolist()

                    
                    #print(best_model_params)
                    
                    #top_model_params_in_cv = best_model_params_across_cvs[f'cv_rep_{cv_repeat_num}']
                    
                    ## RUN CV & TRAIN FINAL MODEL WITH MODEL PARAMETERS SELECTED IN PREVIOUS STEP

                    #for n in range(num_of_top_models_per_cv):
                    training_results['cv_results'][f'cv_rep_{cv_repeat_num}']['model']={}
                    for n in tqdm(range(num_of_top_models_per_cv),desc="Processing", unit="model"):
     
                       

                        #print(f'training best model no. {n+1}')   
                        ## If LR, set l1_ratio to 1 to run L1-penalisation , if XGBoost, extract best performing parameter configuration
                        if model_name=='LogisticRegression':
                            best_model_params={'l1_ratio':1.0}
                        
                        if model_name!='LogisticRegression':
                            #best_model_params=top_model_params_in_cv[n]
                            best_model_params = best_model_params_across_cvs[f'cv_rep_{cv_repeat_num}'][n]
                            
                        #print(best_model_params)
                        model,cv_roc_auc_scores,label_weights_dict = calc_roc_auc_score_of_model(model_name,
                                                                                                 X_train,
                                                                                                 y_train,
                                                                                                 train_params['k_folds'],
                                                                                                 train_params['random_state'],
                                                                                                 outcome_label,
                                                                                                 best_model_params,
                                                                                                 train_params['weight_by_label_freq'],
                                                                                                 train_params)
                        
                        ## Save the best model's ROC-AUC scores as CV validation ROC-AUC
                        #. + Savel 'label_weights_dict' ==> it depends on the train-test split, so label_weights_dict is the same for all
                        #. num_of_top_models_per_cv models
                        if n==0:
                            training_results['cv_results'][f'cv_rep_{cv_repeat_num}']['cv_roc_auc_scores']=cv_roc_auc_scores 
                            training_results['cv_results'][f'cv_rep_{cv_repeat_num}']['label_weights_dict']=label_weights_dict
                            
                        
                        training_results['cv_results'][f'cv_rep_{cv_repeat_num}']['model'][n]=model


                ## Save training results
                #fn=f'../data/{data_param_key}_{model_name}_{training_data_type}_training_results.pickle'
                fn=f'../data/{data_param_key}_{model_name}_{period_end_day}_days_{training_data_type}_training_results.pickle'
                with open(fn, 'wb') as handle:
                    pickle.dump(training_results, handle)
        
loop_time=time.time()
print('Training duration:')
print_elapsed_time(start,loop_time) 
                
                    
                    

