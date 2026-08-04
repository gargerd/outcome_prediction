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

from s9_9_baseline_survival_models_functions import *




### EXTRACT PARAMETERS FOR PARAMETER SEARCH FROM ARGPARSE
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", help="One of the dataset names from the keys of parameters_for_analysis dictionar (see below)")
parser.add_argument("--training_data_type", help="One of the 2 data inclusion types: ['all_days_in_period','last_therapy_day']]")
parser.add_argument("--model", help="One of the 2 models: ['XGBoost','CoxnetSurvival']")
parser.add_argument("--period_end_day", help="One of the 7 periods: ['baseline',31,62,93,125,160,'all']")
parser.add_argument("--ther_arm_duration", help="One of the ther_arm_durationsa: ['4-month','6-month']")
parser.add_argument("--time_origin", help="Time origin, start or end of therapy: ['SOT','EOT']")
#parser.add_argument("--overwrite_existing_params", help="If set to True, overwrites the existing dictionaries with the parameter search results")
#parser.add_argument("--cpu_cores", help="List of integers, setting which CPU cores to be used. i.e. for the first 4 CPU cores: 0-3")


args = parser.parse_args()
dataset_name_=[args.dataset_name]
training_data_type_ = [args.training_data_type]
model_names_ = [args.model]
period_end_day_=[args.period_end_day if args.period_end_day in ['baseline','all'] else int(args.period_end_day)]
ther_arm_durations_ = [args.ther_arm_duration]
time_origin_ = [args.time_origin]
#overwrite_existing_params=args.overwrite_existing_params
#print(overwrite_existing_params)



parameters_for_analysis={'tb21_22_2984_pats_22_vars_result_at_end_of_treatment':{
                            'fn':'tb21_22_2984_pats_22_vars_result_at_end_of_treatment',
                            'result_cat':'RESULT_AT_END_OF_TREATMENT'},
            
                        'tb21_22_2984_pats_22_vars_relapse':{
                            'fn':'tb21_22_2984_pats_22_vars_result_at_end_of_treatment',
                            'survival':True,
                            'result_cat':'RELAPSE'}, 

                         'tb21_22_2263_pats_24_vars_relapse':{
                            'fn':'tb21_22_2263_pats_24_vars_relapse',
                              'survival':True,
                            'result_cat':'RELAPSE'},

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



model_names=['XGBoost','CoxnetSurvival','LogisticHazard'][:]

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
## Define training parameters
train_params={'num_cv_repeats':25,
                'k_folds':5,             
              'weight_by_label_freq':True,
               'label_weights':[1,1],## [index_0: weight for label 0 (negative),index_1: weight for label 1 (positive)], only
                                     ## only considered if weight_by_label_freq=False !
              'random_state':42,
              'test_size_ratio':0.2,
              'early_stopping': True,
                'patience': 10,
                'min_delta': 0.0,
                'val_fraction': 0.2,
              'label2id':label2id}


## Load dataframe containing the last day of drug regimen for each patient
last_initial_therapy_day_df=pd.read_csv('../data/out_last_initial_therapy_day_list_1018_20_21_22_30.csv.gz',index_col=0)
last_initial_therapy_day_df=last_initial_therapy_day_df.set_index('USUBJID')

## Laod pats with relapse df
pats_with_relapse_df=extract_21_22_relapse_pats()




###=========================================================================================
# 2. Calcualte SHAP-values
import time    
start=time.time()
import warnings
warnings.filterwarnings("ignore")

import shap

### CALCULATE SHAP-SCORES (FEATURE IMPORTANCE) ####
def calculate_shap_values(X_train,X_test,model,model_name):

    
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

    elif model_name in ['CoxnetSurvival']:
        predict_fn  = model.predict
        
        background  = shap.sample(X_train, 50)
        explainer   = shap.KernelExplainer(predict_fn, data=background)
        raw_shap    = explainer.shap_values(X_test, silent=True)
    
        # Wrap in Explanation object for consistency with TreeExplainer output
        shap_values = shap.Explanation(
            values          = raw_shap,
            base_values     = np.full(len(X_test), explainer.expected_value),
            data            = X_test.values if hasattr(X_test, 'values') else X_test,
            feature_names   = X_train.columns.tolist() if hasattr(X_train, 'columns') else None
        )

    return shap_values




###==============


def rebuild_pycox_model(model_dict):

    import torch
    import inspect
    import torchtuples as tt
    from pycox.models import CoxPH, DeepHitSingle, LogisticHazard

    model_name = model_dict["model_name"]
    model_params = model_dict["model_params"]
    input_dim = model_dict["input_dim"]
    labtrans = model_dict["labtrans"]
    state_dict = model_dict["state_dict"]

    # Keep only arguments that MLPVanilla actually accepts
    mlp_signature = inspect.signature(tt.practical.MLPVanilla.__init__)
    allowed_mlp_args = set(mlp_signature.parameters.keys()) - {"self"}

    #print('allowed_mlp_args',allowed_mlp_args)

    net_params = {
        k: v for k, v in model_params.items()
        if k in allowed_mlp_args
    }

    # Ensure output dimension is correct for discrete-time models
    if model_name in ["DeepHitSingle", "LogisticHazard"]:
        net_params["out_features"] = labtrans.out_features

    # Rebuild network
    net = tt.practical.MLPVanilla(
        in_features=input_dim,
        num_nodes=model_params['hidden_nodes'],
        output_bias=False,
        **net_params
    )

    # Rebuild pycox model
    if model_name == "CoxPH":
        model = CoxPH(net, tt.optim.Adam)
    elif model_name == "DeepHitSingle":
        model = DeepHitSingle(net, tt.optim.Adam, duration_index=labtrans.cuts)
    elif model_name == "LogisticHazard":
        model = LogisticHazard(net, tt.optim.Adam, duration_index=labtrans.cuts)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    # Load weights
    model.net.load_state_dict(state_dict)

    # Restore labtrans
    model.labtrans = labtrans

    # Eval mode
    model.net.eval()

    return model

###+===============

def calculate_shap_values_logistic_hazard_by_bin(
    X_train,
    X_test,
    model,
    output_type="risk",   # "hazard", "survival", or "risk"
    background_size=100,
    nsamples="auto",
    return_global_importance=True
):
    """
    Calculate SHAP values for a pycox LogisticHazard model separately for each time bin.

    Parameters
    ----------
    X_train : pd.DataFrame or np.ndarray
        Training data used as background/reference set for SHAP.
    X_test : pd.DataFrame or np.ndarray
        Test data for which SHAP values will be computed.
    model : pycox.models.LogisticHazard
        Trained pycox LogisticHazard model.
    output_type : str, default="risk"
        What to explain at each bin:
            - "hazard": discrete hazard at bin k
            - "survival": survival probability at bin k
            - "risk": cumulative event probability at bin k = 1 - survival
    background_size : int, default=50
        Number of background samples drawn from X_train for KernelExplainer.
    nsamples : int or str, default="auto"
        Passed to shap.KernelExplainer.shap_values(..., nsamples=...)
    return_global_importance : bool, default=True
        If True, also returns a dataframe with mean(|SHAP|) per feature per bin.

    Returns
    -------
    shap_results : dict
        Dictionary with one entry per bin:
            shap_results["bin_0"], shap_results["bin_1"], ...
        Each value is a shap.Explanation object.
    global_importance_df : pd.DataFrame, optional
        Rows = features, columns = bins, values = mean absolute SHAP value.
        Only returned if return_global_importance=True.
    """

    # ----------------------------
    # Convert inputs to consistent format
    # ----------------------------
    if isinstance(X_train, pd.DataFrame):
        feature_names = X_train.columns.tolist()
        X_train_np = X_train.values.astype(np.float32)
    else:
        X_train_np = np.asarray(X_train, dtype=np.float32)
        feature_names = [f"feature_{i}" for i in range(X_train_np.shape[1])]

    if isinstance(X_test, pd.DataFrame):
        X_test_np = X_test.values.astype(np.float32)
    else:
        X_test_np = np.asarray(X_test, dtype=np.float32)

    # Background dataset for KernelExplainer
    if background_size < len(X_train_np):
        background = shap.sample(X_train_np, background_size, random_state=0)
    else:
        background = X_train_np.copy()

    # ----------------------------
    # Get model outputs once
    # ----------------------------
    # Hazard shape should be: (n_samples, n_bins)
    hazard_test = model.predict_hazard(X_test_np)

    # Convert to numpy if needed
    if hasattr(hazard_test, "detach"):
        hazard_test = hazard_test.detach().cpu().numpy()
    else:
        hazard_test = np.asarray(hazard_test)

    n_bins = hazard_test.shape[1]

    # Try to recover actual time cut points
    if hasattr(model, "duration_index") and model.duration_index is not None:
        bin_labels = list(model.duration_index)
    else:
        bin_labels = list(range(n_bins))

    shap_results = {}
    global_importance = {}

    # ----------------------------
    # Helper prediction wrappers
    # ----------------------------
    def predict_hazard_bin(X, bin_idx):
        pred = model.predict_hazard(np.asarray(X, dtype=np.float32))
        if hasattr(pred, "detach"):
            pred = pred.detach().cpu().numpy()
        else:
            pred = np.asarray(pred)
        return pred[:, bin_idx]

    def predict_survival_bin(X, bin_idx):
        pred = model.predict_surv_df(np.asarray(X, dtype=np.float32))
        # predict_surv_df usually returns DataFrame with rows=time bins, cols=samples
        if isinstance(pred, pd.DataFrame):
            pred = pred.T.values
        else:
            pred = np.asarray(pred)
            # If shape is (n_bins, n_samples), transpose it
            if pred.shape[0] == n_bins and pred.shape[1] == len(X):
                pred = pred.T
        return pred[:, bin_idx]

    def predict_risk_bin(X, bin_idx):
        surv = model.predict_surv_df(np.asarray(X, dtype=np.float32))
        if isinstance(surv, pd.DataFrame):
            surv = surv.T.values
        else:
            surv = np.asarray(surv)
            if surv.shape[0] == n_bins and surv.shape[1] == len(X):
                surv = surv.T
        return 1.0 - surv[:, bin_idx]

    if output_type == "hazard":
        predict_wrapper_factory = predict_hazard_bin
    elif output_type == "survival":
        predict_wrapper_factory = predict_survival_bin
    elif output_type == "risk":
        predict_wrapper_factory = predict_risk_bin
    else:
        raise ValueError("output_type must be one of: 'hazard', 'survival', 'risk'")

    # ----------------------------
    # Run SHAP separately for each bin
    # ----------------------------
    #for bin_idx in range(n_bins):
    for bin_idx in tqdm(range(n_bins)):        

        def predict_fn(X, b=bin_idx):
            return predict_wrapper_factory(X, b)

        explainer = shap.KernelExplainer(predict_fn, background)
        raw_shap = explainer.shap_values(X_test_np, nsamples=nsamples, silent=True)

        expected_value = explainer.expected_value
        if np.isscalar(expected_value):
            base_values = np.full(X_test_np.shape[0], expected_value)
        else:
            base_values = np.asarray(expected_value)

        bin_key = f"bin_{bin_idx}"
        bin_label = bin_labels[bin_idx]

        shap_exp = shap.Explanation(
            values=raw_shap,
            base_values=base_values,
            data=X_test_np,
            feature_names=feature_names
        )

        shap_results[bin_key] = {
            "bin_index": bin_idx,
            "bin_label": bin_label,
            "output_type": output_type,
            "shap_values": shap_exp
        }

        if return_global_importance:
            global_importance[bin_key] = np.mean(np.abs(raw_shap), axis=0)

    # ----------------------------
    # Global importance dataframe
    # ----------------------------
    if return_global_importance:
        global_importance_df = pd.DataFrame(global_importance, index=feature_names)
        return shap_results, global_importance_df

    return shap_results






#columns_to_drop=['ARM','STUDYID','DAY','index']

### loop over ML models, train them & training results in a dictionary
for data_param_key in dataset_name_:
    print('===========\n',data_param_key,'\n')
    
    #outcome_label=data_param_key.split('vars_')[-1].upper()
    outcome_label = parameters_for_analysis[data_param_key]['result_cat']

    ## LOAD FINAL PATIENT IDS FOR ANALYSIS, SAVED DURING PREPROCESSING OF THE BASELINE MODELS IN NOTEBOOK S9_9
    survival_anal_dir_ = f'../data/survival_analysis'
    if 'pat_ids_fn' in parameters_for_analysis[data_param_key].keys():
        #fn=f"../data/{parameters_for_analysis[data_param_key]['pat_ids_fn']}_final_pat_ids_for_analysis.pickle"
        fn=os.path.join(survival_anal_dir_,
                    f"{parameters_for_analysis[data_param_key]['pat_ids_fn']}_final_pat_ids_for_analysis.pickle")
    else:  
        #fn=f'../data/{data_param_key}_final_pat_ids_for_analysis.pickle'
        fn=os.path.join(survival_anal_dir_,
                    f'{data_param_key}_final_pat_ids_for_analysis.pickle')
    with open(fn, 'rb') as handle:
        final_pat_ids_for_analysis=pickle.load(handle)
        

    ## Load preprocessed-imputed data, and modify the variables (add or drop) depending on the prediction setup, which is contained at the 
    #. end of the "data_param_key" variable
    X,race_colnames = load_and_modify_preprocessed_data(data_param_key)


    for time_origin in time_origin_:


        ## Return dataframe with the outcome label
        pat_ids,y,target_df,outcome_label = return_predict_label_dataframe(parameters_for_analysis,data_param_key,X,
                                                                      outcome_df,outcome_label,model_names,time_origin)
      
       
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
            temp_cols_to_keep=['dr_reg_study_drugs_cumul','vs_Height_STD_NUM_RESULT',
                               'vs_BMI_STD_NUM_RESULT','mb_LJ-culture_CULTURE_STATUS'][:] \
                                + race_colnames #\
                                #+ ae_cumul_colnames
                                #+ arm_cumul_colnames \
            
            for temp_data_name in temporal_data_names:
                cols_to_keep=select_temporal_cols_with_suff_pat_data(temp_data_name,X_subset,temp_col_threshold)
                temp_cols_to_keep.extend(cols_to_keep)
            
            for training_data_type in training_data_type_:
                print(f'Running training with {training_data_type} model')
    
                for ther_arm_dur in ther_arm_durations_:
    
                    if '4-month' in ther_arm_dur and period_end_day in [160,'all']:
                        print(f'Skipping {ther_arm_dur} - cohort at timepoint {period_end_day}')
                        continue
    
    
                    for model_name in model_names_:
                        print(model_name)
                        
                        
                        #fn=f'../data/{data_param_key}_{model_name}_{period_end_day}_days_{training_data_type}_training_results.pickle'
                        fn=os.path.join(survival_anal_dir_,
                                        f'{data_param_key}_{model_name}_{period_end_day}_days_{training_data_type}_{ther_arm_dur}_{time_origin}_training_results.pickle')
    
                        with open(fn, 'rb') as handle:
                            training_results=pickle.load(handle)
        
                        cv_results=training_results['cv_results']
        
                        print(f'{data_param_key} - {period_end_day} - {training_data_type} - {model_name}')
    
    
                        
                        #for cv_repeat_num in range(len([*cv_results])):
                        for cv_repeat_num in tqdm(range(len(cv_results))):
                        #for cv_repeat_num in tqdm(range(1)):
                            #print(f'Calculating SHAP-values for model {model_name},CV-repeat number:{cv_repeat_num}')
        
                            ## Based on the saved random state used at training, re-create the train-test data split
                            rand_state=cv_results[f'cv_rep_{cv_repeat_num}']['rand_state']
        
                            if len(columns_to_drop)>0:
                                columns_to_drop_=columns_to_drop + X_subset.columns[X_subset.columns.str.startswith(tuple(temp_cols_to_drop))].tolist()\
                                                #+ ['dr_reg_study_drugs_cumul']
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
                                X_subset_ = X_subset_.loc[:,~X_subset_.columns.str.startswith('ARM_')]
    
                            
                            X_subset__ = X_subset_[X_subset_['therapy_arm_duration']==ther_arm_dur].drop(columns=['therapy_arm_duration'])
                            
                            X_train,X_test,y_train,y_test,\
                                            X_train_pat_ids,X_test_pat_ids= create_std_training_testing_data(X_subset__,
                                                                                                             y,
                                                                                                             pat_ids_,
                                                                                                             train_params['test_size_ratio'],
                                                                                                             rand_state,training_data_type,
                                                                                                             columns_to_drop_,
                                                                                                             period_end_day,
                                                                                                             outcome_label,
                                                                                                             cv_repeat_num=cv_repeat_num,
                                                                                                             final_pat_ids_for_analysis=final_pat_ids_for_analysis)
        
                            
        
                            X_train,X_test = scale_by_training_data(X_train, X_test)
                            #print(y)
        
                            
                            ## Calculate SHAP-values * prediction probabilities for both training and testing data
                            for X_data,y_data,prefix in zip([X_train,X_test][:],[y_train,y_test][:],['train','test'][:]):
                            
                                if isinstance(cv_results[f'cv_rep_{cv_repeat_num}']['model'], dict):
    
                                    if is_pycox_model(model_name)==True:
                                        model = rebuild_pycox_model(cv_results[f'cv_rep_{cv_repeat_num}']['model'][0])
                                        shap_results, global_importance_df = calculate_shap_values_logistic_hazard_by_bin(
                                                                                                        X_train=X_train,
                                                                                                        X_test=X_data,
                                                                                                        model=model,
                                                                                                        output_type="risk",   # recommended for interpretation
                                                                                                        background_size=100,
                                                                                                        nsamples="auto"
                                                                                                    )
                                        pred_surv = model.predict_surv_df(np.asarray(X_data, dtype=np.float32))
                                        pred_risk = model.predict_hazard(np.asarray(X_data, dtype=np.float32))
                                        pred_surv.columns = X_data.index.tolist()
                                        pred_risk = pd.DataFrame(index=X_data.index.tolist(),columns=pred_surv.index,data=pred_risk)
                                        #pred_risk = pred_risk.T
                                        pred_surv = pred_surv.T

                                        cv_results[f'cv_rep_{cv_repeat_num}'][f'{prefix}_pred_risk']=pred_risk
                                        cv_results[f'cv_rep_{cv_repeat_num}'][f'{prefix}_pred_surv']=pred_surv
                                        cv_results[f'cv_rep_{cv_repeat_num}'][f'{prefix}_true_labels']=y_data.values
                                        cv_results[f'cv_rep_{cv_repeat_num}'][f'{prefix}_shap_values']=shap_results
                                        
                                    
                                    if is_pycox_model(model_name)==False:
                                        model=cv_results[f'cv_rep_{cv_repeat_num}']['model'][0]       
                                
                                        shap_values=calculate_shap_values(X_train,X_data,model,model_name)
                                        cv_results[f'cv_rep_{cv_repeat_num}'][f'{prefix}_shap_values']=shap_values
        
        
                                        #pred_prob = model.predict_proba(X_data)[:, 1]
                                        pred_risk = model.predict(X_data)
                                        
                                        cv_results[f'cv_rep_{cv_repeat_num}'][f'{prefix}_pred_risk']=pred_risk
                                        cv_results[f'cv_rep_{cv_repeat_num}'][f'{prefix}_true_labels']=y_data.values
                                
                            
                           
                             
                        training_results['cv_results']=cv_results
                
                        #fn=f'../data/{data_param_key}_{model_name}_{period_end_day}_days_{training_data_type}_training_results.pickle'
                        fn=os.path.join(survival_anal_dir_,
                                        f'{data_param_key}_{model_name}_{period_end_day}_days_{training_data_type}_{ther_arm_dur}_{time_origin}_training_results.pickle')
                        with open(fn, 'wb') as handle:
                            pickle.dump(training_results, handle)
        
loop_time=time.time()
print('Training duration:')
print_elapsed_time(start,loop_time) 
                
                    
                    

