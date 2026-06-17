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

                         'tb20_21_22_2908_pats_7_vars_relapse':{
                            'result_cat':'RELAPSE',
                            'fn':'tb20_21_22_2908_pats_7_vars_relapse',
                           'include_rifaquin':True},

                        'tb20_21_22_2908_pats_7_vars_relapse_ext_pats':{
                            'result_cat':'RELAPSE',
                            'fn':'tb20_21_22_2908_pats_7_vars_relapse',
                           'include_rifaquin':True},

                        'tb20_21_22_2905_pats_8_vars_relapse_ext_pats':{
                              'result_cat':'RELAPSE',
                            'fn':'tb20_21_22_2905_pats_8_vars_relapse',
                           'include_rifaquin':True},

                         'tb20_21_22_2905_pats_8_vars_relapse_ext_pats_no_rifaquin':{
                              'result_cat':'RELAPSE',
                            'fn':'tb20_21_22_2905_pats_8_vars_relapse',
                             #'pat_ids_fn':'tb21_22_2984_pats_22_vars_relapse_ext_pats',
                           'include_rifaquin':False},

                         'tb20_21_22_2905_pats_8_vars_relapse_without_dr_reg_ext_pats_no_rifaquin':{
                              'result_cat':'RELAPSE',
                            'fn':'tb20_21_22_2905_pats_8_vars_relapse',
                             'pat_ids_fn':'tb20_21_22_2905_pats_8_vars_relapse',
                           'include_rifaquin':False},

                        'tb21_22_2984_pats_22_vars_result_at_end_of_treatment_without_dr_reg':{
                            'fn':'tb21_22_2984_pats_22_vars_result_at_end_of_treatment',
                             'pat_ids_fn':'tb21_22_2984_pats_22_vars_result_at_end_of_treatment',
                            'result_cat':'RESULT_AT_END_OF_TREATMENT'},


                        'tb21_22_2984_pats_22_vars_result_at_end_of_treatment_without_dr_reg_with_arm':{
                            'fn':'tb21_22_2984_pats_22_vars_result_at_end_of_treatment',
                             'pat_ids_fn':'tb21_22_2984_pats_22_vars_result_at_end_of_treatment',
                            'corr_test_for_arm':True,
                            'result_cat':'RESULT_AT_END_OF_TREATMENT'},

                          'tb21_22_2984_pats_22_vars_relapse_ext_pats_without_dr_reg_with_arm':{
                            'fn':'tb21_22_2984_pats_22_vars_result_at_end_of_treatment',
                            'pat_ids_fn':'tb21_22_2984_pats_22_vars_relapse_ext_pats',
                            'corr_test_for_arm':True,
                            'result_cat':'RELAPSE'}, 

                         ### train on one/more studies, validate on completely held out study
                          'tb21_22_2984_pats_22_vars_result_at_end_of_treatment_without_dr_reg_train_tb21':{
                            'fn':'tb21_22_2984_pats_22_vars_result_at_end_of_treatment',
                             #'pat_ids_fn':'tb21_22_2984_pats_22_vars_result_at_end_of_treatment',
                            'corr_test_for_arm':True,
                            'training_cohort':['TB-1021'],
                            'validation_cohort':['TB-1022'],
                            'result_cat':'RESULT_AT_END_OF_TREATMENT'},
                            
                            'tb21_22_2984_pats_22_vars_result_at_end_of_treatment_without_dr_reg_train_tb22':{
                            'fn':'tb21_22_2984_pats_22_vars_result_at_end_of_treatment',
                             #'pat_ids_fn':'tb21_22_2984_pats_22_vars_result_at_end_of_treatment',
                            'corr_test_for_arm':True,
                            'training_cohort':['TB-1022'],
                            'validation_cohort':['TB-1021'],
                            'result_cat':'RESULT_AT_END_OF_TREATMENT'},
                            
                            'tb21_22_2984_pats_22_vars_relapse_ext_pats_without_dr_reg_train_tb21':{
                            'fn':'tb21_22_2984_pats_22_vars_result_at_end_of_treatment',
                            #'pat_ids_fn':'tb21_22_2984_pats_22_vars_relapse_ext_pats',
                            'corr_test_for_arm':True,
                            'training_cohort':['TB-1021'],
                            #'include_rifaquin':True,
                            'validation_cohort':['TB-1022'], # ['TB-1022']
                            'result_cat':'RELAPSE'}, 
                            
                            'tb21_22_2984_pats_22_vars_relapse_ext_pats_without_dr_reg_train_tb22':{
                            'fn':'tb21_22_2984_pats_22_vars_result_at_end_of_treatment',
                            #'pat_ids_fn':'tb21_22_2984_pats_22_vars_relapse_ext_pats',
                            'corr_test_for_arm':True,
                            'training_cohort':['TB-1022'],
                            'validation_cohort':['TB-1021'],# ['TB-1021']
                            #'include_rifaquin':True,
                            'result_cat':'RELAPSE'}, 

                         'tb20_21_22_2905_pats_8_vars_relapse_without_dr_reg_ext_pats_train_tb21':{
                              'result_cat':'RELAPSE',
                            'fn':'tb20_21_22_2905_pats_8_vars_relapse',
                             'corr_test_for_arm':True,
                             #'pat_ids_fn':'tb20_21_22_2905_pats_8_vars_relapse_ext_pats',
                             'training_cohort':['TB-1021'],
                            'validation_cohort':['TB-1020'],                         
                           'include_rifaquin':True},
                         
                         'tb20_21_22_2905_pats_8_vars_relapse_without_dr_reg_ext_pats_train_tb22':{
                              'result_cat':'RELAPSE',
                            'fn':'tb20_21_22_2905_pats_8_vars_relapse',
                             'corr_test_for_arm':True,
                             #'pat_ids_fn':'tb20_21_22_2905_pats_8_vars_relapse_ext_pats',
                             'training_cohort':['TB-1022'],
                            'validation_cohort':['TB-1020'],                         
                           'include_rifaquin':True},

                         'tb20_21_22_2905_pats_8_vars_relapse_without_dr_reg_ext_pats_train_tb21_tb22':{
                              'result_cat':'RELAPSE',
                            'fn':'tb20_21_22_2905_pats_8_vars_relapse',
                             'corr_test_for_arm':True,
                             #'pat_ids_fn':'tb20_21_22_2905_pats_8_vars_relapse_ext_pats',
                             'training_cohort':['TB-1022','TB-1021'],
                            'validation_cohort':['TB-1020'],                         
                           'include_rifaquin':True},







                         

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
pats_with_relapse_df=extract_21_22_relapse_pats(include_rifaquin=True,
                                               extended_pats=True)




###=========================================================================================
# 2. Calcualte SHAP-values
import time    
start=time.time()
import warnings
warnings.filterwarnings("ignore")


## CALCULATE SHAP-SCORES (FEATURE IMPORTANCE) ####
def calculate_shap_values(X_train,X_test,model,model_name):
    import shap
    
    #importances_dict[training_method][train_ds_type+'_train_data'][model_type_name]={}

    if model_name in ['RandomForest','GradientBoost','XGBoost']:
        explainer=shap.TreeExplainer(model,data=X_train)
        shap_values=explainer(X_test)

    elif model_name in ['LogisticRegression','Lasso','Ridge','ElasticNet']:
        #explainer=shap.LinearExplainer(model,data=X_train,masker=shap.maskers.Impute(data=X_train))
        #shap_values=explainer(X_test)

        #masker = shap.maskers.Independent(X_train)
        masker = shap.maskers.Impute(X_train)
        explainer = shap.Explainer(model, masker=masker, algorithm="linear")
        shap_values = explainer(X_test)

    elif model_name in ['DenseNetwork']:
        explainer=shap.DeepExplainer(model,data=torch.tensor(X_train.values,dtype=torch.float32))
        shap_values=explainer.shap_values(torch.tensor(X_test.values,dtype=torch.float32))
        
    elif model_name in ['SVR','KNN']:
        explainer=shap.KernelExplainer(model.predict, data=shap.kmeans(X_train, 1),
                                        masker=shap.maskers.Independent(X_train, max_samples=100))
        shap_values=explainer.shap_values(X_test)                                                       

    return shap_values



####=============
def return_additional_exlcuded_pats(parameters_for_analysis,
                                   data_param_key,
                                   d_1022):

    fn='../data/tb21_22_2984_pats_22_vars_result_at_end_of_treatment_preproc_data_with_imp.csv.gz'
    X_subset=pd.read_csv(fn,index_col=0)
    X_subset=X_subset.rename(columns=lambda x: x.replace('<', 'lower than'))
    X_subset=X_subset.rename(columns=lambda x: x.replace('>', 'higher than'))

    outcome_tb1021 =pd.read_csv('../data/tb_1021_outcome.csv.gz',index_col=0)
    outcome_tb1021 = outcome_tb1021.loc[X_subset[X_subset['STUDYID']=='TB-1021']['USUBJID'].unique(),:]
    oc_1021 = outcome_tb1021[outcome_tb1021['RESULT_AT_END_OF_TREATMENT']=='FAVOURABLE'].replace('UNASSESSABLE',np.nan)
    
    
    outcome_tb1022 = pd.read_csv('../data/tb_1022_outcome.csv.gz',index_col=0)
    outcome_tb1022 = outcome_tb1022.loc[X_subset[X_subset['STUDYID']=='TB-1022']['USUBJID'].unique(),:]
    outcome_tb1022['RESULT_AT_END_OF_TREATMENT']#.value_counts(dropna=False)
    oc_1022 = outcome_tb1022[outcome_tb1022['RESULT_AT_END_OF_TREATMENT']=='FAVOURABLE'].replace('UNASSESSABLE',np.nan)
    
    oc_1022 = oc_1022.replace('NOT ASSESSABLE',np.nan)

    oc = pd.concat([outcome_tb1021,outcome_tb1022],axis=0)

    ## LOAD FINAL PATIENTS CONSIDERED FOR EOT OUTCOME AND RELAPSE PREDICTION

    #rel = ddd[[*parameters_for_analysis][2]]
    #eot = ddd[[*parameters_for_analysis][0]]

    fn=f'../data/{[*parameters_for_analysis][0]}_max_num_of_pats.csv'
    eot = pd.read_csv(fn,index_col=0)
    
    fn=f'../data/{data_param_key}_max_num_of_pats.csv'
    rel = pd.read_csv(fn,index_col=0)
    
    
    ## The number of inlcuded patients changed over the therapy periods, due to sparsity and therapy duration (4 or 6-month)
    #. ==> in s9_3_baseline_ML_models notebook (Section: "Number of patients across study/arm/outcome..."), dataframes from the timepoint with the highest number of included patients were saved
    #. ==> These include the final patients considered for analysis, and they contain the highest number of patients included for prediction per prediction label, 
    #. ==> There us a subset of patients with favourable EOT outcome, who were exlcuded from relapse prediction, due to some reasons (lost- to FU, death,...)
    #. ==> Take these patients and extract the reason for exclusion (most of them were already extracted in the functions return_fav_unfav_pats_tb_1021/1022, however not all)
    
    pats_excl_from_relapse = list(set(eot.loc[eot['RESULT_AT_END_OF_TREATMENT']=='Favourable','USUBJID']) - set(rel.loc[:,'USUBJID']))
    
    
    ## In the original extraction algorithm, not all patients were captured with reason for exclusion 
    #. ==> Collect those whoe were excluded with corresponding reason for exclusion
    excl_1022_pats_incomplete=[]
    for k in d_1022.keys():
        #print(k,len(d_1022[k]['USUBJID'].tolist()))
        excl_1022_pats_incomplete.extend(d_1022[k]['USUBJID'].tolist())
        #print(len(d_1022[k]))
    
    ##. ==== TB-1022 (OFLOTUB) ======
    ## Extract all patients who were excluded from relapse prediction 
    all_1022_pats_exluded_from_relapse = oc.loc[(oc.index.isin(pats_excl_from_relapse))&\
                                                (oc['STUDYID']=='TB-1022'),:].index.tolist()
    
    ## Extrcat patients who were exlcuded from relapse prediction, but were not captured by the return_fav_unfav_pats_tb_1022 function
    additional_excluded_1022_pats = list(set(all_1022_pats_exluded_from_relapse) - set(excl_1022_pats_incomplete))
    
    
    additional_excluded_1022_pats = (oc.loc[additional_excluded_1022_pats,:]
                                     .fillna('Unknown_unfavourable_cat_at_24_months')
                                     .groupby('UNFAVOURABLE_OUTCOME_CATEGORY_AT_24_MONTHS')
                                     .apply(lambda x: x.index.tolist()).to_dict())
    
    
    ##. ==== TB-1021 (REMOXTB) ======
    all_1021_pats_exluded_from_relapse = oc.loc[(oc.index.isin(pats_excl_from_relapse))&\
                                                (oc['STUDYID']=='TB-1021'),:].index.tolist()
    
    
    oc_=oc.copy()
    
    ## MOST OF THESE PATIENTS HAVE A FAVOURABLE OUTCOME AT EOT AND MONTH 18, BUT THERE IS UNFAVOURABLE AT LEAST ONCE IN BETWEEN 
    
    # FOR 10 PATIENTS, FAVOURABLE IS THE UNFAVOUR. OUTCOME CATEGORY ==> CHANGE THIS TO REFLECT THE PREVIOUSLY DESCRIBED CASE
    oc_.loc[(oc_.index.isin(all_1021_pats_exluded_from_relapse))&\
            (oc_['UNFAVOURABLE_OUTCOME_CATEGORY_AT_18_MONTHS']=='FAVOURABLE'),'UNFAVOURABLE_OUTCOME_CATEGORY_AT_18_MONTHS']='favour_at_EOT_and_18_months_unfavour_between'
    
    # FOR PATIENTS WITH NAN AS THE UNFAVOUR. OUTCOME CATEGORY ==> CHANGE THIS TO REFLECT THE PREVIOUSLY DESCRIBED CASE
    additional_excluded_1021_pats = (oc_.loc[all_1021_pats_exluded_from_relapse,:]
                                     .fillna('favour_at_EOT_and_18_months_unfavour_between')
                                     .groupby('UNFAVOURABLE_OUTCOME_CATEGORY_AT_18_MONTHS')
                                     .apply(lambda x: x.index.tolist()).to_dict())

    
    return additional_excluded_1022_pats, additional_excluded_1021_pats


####======
## Return preprocessed data of patients who were exlcuded in relapse prediction due to various reasons (mostly lost to follow-up)
def return_excluded_pats_data(X,period_end_day,training_data_type,data_param_key,parameters_for_analysis):

    ## LOAD DATAFRAME CONTAINING PREPROCESSED DATA FOR ALL PATIENTS
    data_dir_ = f'../data/raw_data_for_analysis/{[*parameters_for_analysis][0]}'
    os.makedirs(data_dir_,exist_ok=True)
    
    #print('X_full_',X_full_.shape)
    pat_id_name= 'pat_ids_all'
    #pat_id_name= 'pat_ids_at_period'
    fn=os.path.join(data_dir_,
                   f"{period_end_day}_days_{training_data_type}_{pat_id_name}.pickle")
    #X_full_=pd.read_csv(fn,index_col=0)
    with open(fn, 'rb') as handle:
        d= pickle.load(handle)

    X_full = d['X_full']
    y_full = d['y_full']
    
    ## EXTRACT PATIENTS NOT INCLUDED IN RELAPSE PREEDICTION DUE TO VARIOUS REASONS
    _,_, d_1022 = return_fav_unfav_pats_tb_1022(X)
    _,_, d_1021 = return_fav_unfav_pats_tb_1021(X)
    
    l=[]
    for k,v in d_1022.items():
        v['reason']=k
        l.append(v)

    ## IF NOT EXTENDED PATIENTS WERE INCLUDED, ALSO EXTRACT PATIENTS FROM REMOXTB THAT WERE ORIGINALLY EXCLUDED DUE TO MISSING MGIT AT MONTTH 18
    ##. AND MISSING LABEL AT MONTH 12
    if 'ext_pats' not in data_param_key:
        for k,v in d_1021.items():
            v_=pd.DataFrame({'USUBJID':v,'reason':[k,]*len(v)})
            #v_['reason']=k
            l.append(v_)


    ### EXTRACT ADDITIONAL PATIENTS WHO WEREN'T EXTRACTED IN return_fav_unfav_pats_tb_1022 FUNCTIONS + ADD THEM TO THE FINAL DATAFRAME
    additional_excluded_1022_pats, additional_excluded_1021_pats = return_additional_exlcuded_pats(parameters_for_analysis,
                                                                                                       data_param_key,
                                                                                                       d_1022)

    for k,v in additional_excluded_1021_pats.items():
        v_=pd.DataFrame({'USUBJID':v,'reason':[k,]*len(v)})
        #v_['reason']=k
        l.append(v_)

    for k,v in additional_excluded_1022_pats.items():
        v_=pd.DataFrame({'USUBJID':v,'reason':[k,]*len(v)})
        #v_['reason']=k
        l.append(v_)
        
    excl_pats_df = pd.concat(l)[['USUBJID','reason','DSSTDY']]
    
    comm_idx = list(set(X_full.index) & set(excl_pats_df['USUBJID'].tolist()))
    print('Including',f"{len(comm_idx)} out of {len(excl_pats_df)} excluded patients.")
    
    #X_full_.loc[comm_idx,:]
    #print(y_full)


    return X_full.loc[comm_idx,:], y_full.loc[comm_idx], excl_pats_df







#columns_to_drop=['ARM','STUDYID','DAY','index']

### loop over ML models, train them & training results in a dictionary
for data_param_key in dataset_name_:
    print('===========\n',data_param_key,'\n')
    
    #outcome_label=data_param_key.split('vars_')[-1].upper()
    outcome_label = parameters_for_analysis[data_param_key]['result_cat']

    ## LOAD FINAL PATIENT IDS FOR ANALYSIS, SAVED DURING PREPROCESSING OF THE BASELINE MODELS IN NOTEBOOK S9_3
    final_pat_ids_for_analysis = load_final_patient_for_analysis(parameters_for_analysis=parameters_for_analysis,
                                                                 data_param_key=data_param_key)
        

    ## Load preprocessed-imputed data, and modify the variables (add or drop) depending on the prediction setup, which is contained at the 
    #. end of the "data_param_key" variable
    X,race_colnames = load_and_modify_preprocessed_data(data_param_key)


    ## Return dataframe with the outcome label
    pat_ids,y,target_df,outcome_label = return_predict_label_dataframe(parameters_for_analysis,data_param_key,X,
                                                                  outcome_df,outcome_label,model_names)
        
       
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

        
        #X_subset,ae_cumul_colnames = calucate_cumul_adverse_clinical_events(X_subset,period_end_day)
       

        ## Define baseline columns to keep 
        temp_col_threshold=0.3 # 0.3 
        temporal_data_names=['re','ae','cm','su','mh'][:]
        
        if 'corr_test_for_arm' in parameters_for_analysis[data_param_key] or 'without_dr_reg' in data_param_key:
            temporal_data_names = ['re','ae','cm','su','mh'][:-1]
            
        temp_cols_to_keep=['dr_reg_study_drugs_cumul','vs_Height_STD_NUM_RESULT',
                           'vs_BMI_STD_NUM_RESULT','mb_LJ-culture_CULTURE_STATUS'][:] \
                            + race_colnames #\
                            #+ ae_cumul_colnames
                            #+ arm_cumul_colnames \
                        
        
        for temp_data_name in temporal_data_names:
            cols_to_keep=select_temporal_cols_with_suff_pat_data(temp_data_name,X_subset,temp_col_threshold)
            temp_cols_to_keep.extend(cols_to_keep)

        for training_data_type in training_data_type_:
            
            for n, model_name in enumerate(model_names_[:]):
                print('\n=====================')
                print(model_name)
    
                fn=f'../data/{data_param_key}_{model_name}_{period_end_day}_days_{training_data_type}_training_results.pickle'
                with open(fn, 'rb') as handle:
                    training_results=pickle.load(handle)

            
                cv_results=training_results['cv_results']

                print(f'{data_param_key} - {period_end_day} - {training_data_type} - {model_name}')

                 
                ## IF RELAPSE PREEDICTION, EXTRACT THE PREPROCESED DATA OF PATIENTS WITH FAV. EOT OUTCOME, BUT WERE EXLCUDED DUE TO MISSING RELAPSE LABEL
                ##. ==> VARIOUS REASONS FOR THIS: LOST TO FOLLOW-UP DEATH, ETC.
                if outcome_label=='RELAPSE':
                    X_excluded,y_excluded, excl_pats_df = return_excluded_pats_data(X,
                                                                                    period_end_day,
                                                                                    training_data_type,
                                                                                    data_param_key,
                                                                                   parameters_for_analysis)
                            


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
                    if 'corr_test_for_arm' not in parameters_for_analysis[data_param_key]:
                            X_subset_ = X_subset_.loc[:,~X_subset_.columns.str.startswith('ARM_')]

                ## LOAD DATAFRAME CONTAINING PREPROCESSED DATA FOR  PATIENTS CONSIDERED AT PERIOD
                #data_dir_ = f'../data/raw_data_for_analysis/{[*parameters_for_analysis][0]}'
                data_dir_ = f'../data/raw_data_for_analysis/{data_param_key}'

                ## LOAD FINAL PATIENT IDS FOR ANALYSIS, SAVED DURING PREPROCESSING OF THE BASELINE MODELS IN NOTEBOOK S9_3
                if 'pat_ids_fn' in parameters_for_analysis[data_param_key].keys():
                    data_dir_ = f"../data/raw_data_for_analysis/{parameters_for_analysis[data_param_key]['pat_ids_fn']}"

                else:  
                    data_dir_ = f'../data/raw_data_for_analysis/{data_param_key}'

                
                #print('X_full_',X_full_.shape)
                pat_id_name= 'pat_ids_at_period' #if training_data_type=='last_therapy_day' else 'pat_ids_all'
                fn=os.path.join(data_dir_,
                               f"{period_end_day}_days_{training_data_type}_{pat_id_name}.pickle")
                #X_full_=pd.read_csv(fn,index_col=0)
                with open(fn, 'rb') as handle:
                    d= pickle.load(handle)
            
                X_full = d['X_full']
                y_full = d['y_full']

                ## Drop drug regimen variables, except for drug adherence (dr_cumul_dose)
                if 'without_dr_reg' in data_param_key:
                    X_full = X_full.loc[:,~X_full.columns.str.startswith('dr_reg')]
                    
                        
                #for cv_repeat_num,_ in enumerate(tqdm(range(len([*cv_results])),desc="Processing", unit="model")):
                for cv_repeat_num,_ in enumerate(range(len([*cv_results]))):
                      
                    if isinstance(cv_results[f'cv_rep_{cv_repeat_num}']['model'], dict):
                        model=cv_results[f'cv_rep_{cv_repeat_num}']['model'][0]
                        train_feat_names = model.feature_names_in_


                    rand_state=train_params['random_state'] + cv_repeat_num
                    
                    '''
                    X_train,X_test,y_train,y_test,\
                                X_train_pat_ids,X_test_pat_ids= create_std_training_testing_data(X_subset_,
                                                                                                 y,
                                                                                                 pat_ids_,
                                                                                                 train_params['test_size_ratio'],
                                                                                                 rand_state,training_data_type,
                                                                                                 columns_to_drop_,
                                                                                                 period_end_day,
                                                                                                 outcome_label,
                                                                                                 cv_repeat_num=cv_repeat_num,
                                                                                                 data_param_key=data_param_key,
                                                                                                 parameters_for_analysis = parameters_for_analysis,
                                                                                                 final_pat_ids_for_analysis=final_pat_ids_for_analysis)
                    
                    ''' 
                    
                    #X_train,X_test = scale_by_training_data(X_train, X_test)

                    # print(cv_repeat_num,len(X_train.index),len(X_test.index))

                    X_train_ids=final_pat_ids_for_analysis[period_end_day][cv_repeat_num]['X_train_ids']
                    X_test_ids=final_pat_ids_for_analysis[period_end_day][cv_repeat_num]['X_test_ids']
                    print('X_test_ids',len(X_test_ids))

                    X_train=X_full.loc[list(set(X_full.index)&set(X_train_ids)),:]
                    #X_train=X_full.loc[X_train_ids,:]
                    
                    #X_test=X.loc[list(set(X.index)&set(X_test_ids)),:]
                    X_test=X_full.loc[list(set(X_full.index)&set(X_test_ids)),:]
                    #X_test=X_full.loc[X_test_ids,:]

                    y_train=y.loc[list(set(X_full.index)&set(X_train_ids)),:]
                    y_test=y.loc[list(set(X_full.index)&set(X_test_ids)),:]
                    
                    all_pats = X_train.index.unique().tolist() + X_test.index.unique().tolist()
                    print('X_train',X_train.index.unique().shape[0],'X_test',X_test.index.unique().shape[0])
                    print('all pats:',len(all_pats))
                    
                    #X_train,X_test = scale_by_training_data(X_train, X_test)
                    #all_pats = X_train.index.unique().tolist() + X_test.index.unique().tolist()

      
                    
                    X_train, X_test=scale_by_training_data(X_train, X_test)

                    ## If we include ARM information as well, and  test on held-out studies, the arms of the train and test data are not overlapping completely
                    ##. ==> keep only the column names used for training for both cohorts
                    X_train,X_test = X_train[train_feat_names],X_test[train_feat_names]


                    X_list=[X_train,X_test]
                    y_list=[y_train,y_test]
                    dtype_name_list=['train','test']
                    
                    
                    if outcome_label=='RELAPSE':
                        
                        ## If we include ARM information as well, and  test on held-out studies, the arms of the train and test data are not overlapping completely
                        ##. ==> The excluded patient data  doesn't have arm information, add them with 0 values 
                        X_excluded_ = X_excluded.copy()
                        miss_cols = list(set(train_feat_names) - set(X_excluded_.columns))
                        if len(miss_cols)>0:
                            X_excluded_[miss_cols]=0
                            print(f'Added missing columns: {miss_cols} with values 0')
                            
                        X_excluded_ = X_excluded_.loc[:,X_train.columns].copy()
                        X_excluded_.columns = X_excluded_.columns.astype(str)
                        _,X_excluded_ = scale_by_training_data(X_full.loc[list(set(X_full.index)&set(X_train_ids)),X_train.columns.tolist()],
                                                              X_excluded_)
                        
                        X_list.append(X_excluded_)
                        y_list.append(y_excluded)
                        dtype_name_list.append('excluded')

            
                    ## Calculate SHAP-values * prediction probabilities for both training and testing data
                    #for X_data,y_data,prefix in zip([X_train,X_test][1:],[y_train,y_test][1:],['train','test'][1:]):
                    for X_data,y_data,prefix in zip(X_list,
                                                    y_list,
                                                    dtype_name_list):
                        print(prefix,X_data.shape)
                        
                        shap_values=calculate_shap_values(X_train,X_data,model,model_name)
                        
                        shap_values = pd.DataFrame(index = X_data.index,data = shap_values.values, columns=X_data.columns)
                        cv_results[f'cv_rep_{cv_repeat_num}'][f'{prefix}_shap_values']=shap_values

                        #print(shap_values.values.shape)

                        pred_prob = model.predict_proba(X_data)[:, 1]
                        
                        pred_prob = pd.DataFrame(index = X_data.index,data = pred_prob, columns=[f'{prefix}_pred_prob'])
                        cv_results[f'cv_rep_{cv_repeat_num}'][f'{prefix}_pred_prob']=pred_prob
                        cv_results[f'cv_rep_{cv_repeat_num}'][f'{prefix}_true_labels']=y_data
                        
                        
                training_results['cv_results']=cv_results
        
                fn=f'../data/{data_param_key}_{model_name}_{period_end_day}_days_{training_data_type}_training_results.pickle'
                with open(fn, 'wb') as handle:
                    pickle.dump(training_results, handle)
        
loop_time=time.time()
print('Training duration:')
print_elapsed_time(start,loop_time) 
                
                    
                    

