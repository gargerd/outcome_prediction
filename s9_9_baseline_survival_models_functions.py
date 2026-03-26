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
from sklearn.model_selection import train_test_split

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300


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

#fn='../data/'+parameters_for_analysis[data_param_key]['fn']+'_preproc_data_with_imp.csv.gz'
#X=pd.read_csv(fn,index_col=0)



###===================
def load_merged_data_of_lab_vars():
    #load patient IDs who are considered in this  analysis
    pat_id_df=pd.read_csv('../data/patients_in_analysis.csv.gz',index_col=0)
    # get all pat ids
    all_ids=pat_id_df['USUBJID'].to_list()

    fname='merged_df.csv.gz'
    
    fn=os.path.join('../data/',fname)
    merged_df=pd.read_csv(fn,low_memory=False,index_col=0)

    return merged_df

###===================
def return_race_dict():
    data=load_merged_data_of_lab_vars()
    race_df=data.drop_duplicates(subset=['USUBJID'])[['USUBJID','STUDYID','RACE']]
    race_df.loc[race_df['STUDYID']=='TB-1022','RACE']='BLACK'
    race_dict=race_df.set_index('USUBJID')['RACE'].to_dict()

    return race_dict

## Return a dictionary containing the race of the patients
race_dict=return_race_dict()

##=========================================
## SCALE TRAINING AND TESTING DATA WITH A STANDARD SCALER TRAINED ON THE TRAINING DATA
def scale_by_training_data(X_train, X_test):
    from sklearn.preprocessing import StandardScaler

    ## Standardise non-binary numerical columns ==> select numerical columns and drop columns already scaled
    numerical_cols=X_train.select_dtypes(exclude=['object'])
    non_binary_num_cols = []

    for col in numerical_cols.columns:
        unique_values = np.sort(numerical_cols[col].unique())
        if np.array_equal(unique_values, np.array([0, 1]))==False:
            non_binary_num_cols.append(col)


    if len(non_binary_num_cols)>0:

        #print((X_train==np.inf).any())
        std_scaler = StandardScaler()
        std_scaler.fit(X_train.loc[:,non_binary_num_cols])
        X_train.loc[:,non_binary_num_cols]=std_scaler.transform(X_train.loc[:,non_binary_num_cols])
        X_test.loc[:,non_binary_num_cols]=std_scaler.transform(X_test.loc[:,non_binary_num_cols])

        #X_train.loc[:,non_binary_num_cols]=(StandardScaler().fit_transform(X_train.loc[:,non_binary_num_cols]))

        '''        
        ## As DAY colum,n also gets standardised, transform the training_days number as well, in order to susbet the
        #  data to training-testing set
        std_scaler = StandardScaler()
        std_scaler.fit(X.loc[:, 'DAY'].values.reshape(-1, 1))
        std_training_days = std_scaler.transform([[training_days]])
        '''

    return (X_train, X_test)


##=========================================
## 
def backward_fill_and_extract_vars_at_baseline(X_subset,columns_to_drop):

    ## Extract columns selected for input 
    final_cols=X_subset.drop(columns=columns_to_drop).columns.tolist()

    ## For each final variable, check the number of  patiens who have only missing values in the first month.
    #. If a variable has a high relative missing rate in the first month across the patients (>0.1, i.e. missing for more than 10 % of patients),
    #. don't backfill that variable 
    num_of_pats_with_missing_vars_in_first_month=X_subset[['DAY']+final_cols].sort_values('DAY').groupby('USUBJID').apply(lambda x:x.loc[x['DAY']<31,:].isna().all()).sum().sort_values()
    num_of_pats_with_missing_vars_in_first_month_=num_of_pats_with_missing_vars_in_first_month/X_subset['USUBJID'].unique().shape[0]
    vars_to_backfill = num_of_pats_with_missing_vars_in_first_month_[num_of_pats_with_missing_vars_in_first_month_<=0.05].index.tolist()
    vars_not_to_backfill = num_of_pats_with_missing_vars_in_first_month_[num_of_pats_with_missing_vars_in_first_month_>0.0].index.tolist()


    #print('num_of_pats_with_missing_vars_in_first_month_',num_of_pats_with_missing_vars_in_first_month_[num_of_pats_with_missing_vars_in_first_month_>0])

    ## Drop variables not to backfill from the final cols
    final_cols_=[fin_col for fin_col in final_cols if fin_col in vars_to_backfill]
    #final_cols_=final_cols
    #print('final_cols_',final_cols_)
    
    ## Drop visits (==rows), where there are missing data in the selected variable columns
    a=X_subset[['DAY']+final_cols_].dropna(how='any',subset=final_cols_,axis=0)

    #print(a.columns)
    #print(a)
    
   
    ## Extract the first day of study, where the patient has no missing information in the input variables
    ## ==> this way we can check which was the first visit where all of the selected variables have non-missing measurements 
    #. ==> To impute missing variables at earlier visits, backward fill the missing variable's first valid value
    #. ==> The upper limit where a backward fill is acceptable is 31 days.
    first_complete_day_df=a[['DAY']+final_cols_].sort_values(by=['DAY']).groupby('USUBJID').apply(lambda x: x.loc[x.index[0],:])
    
    pats_with_miss_vars=first_complete_day_df[(first_complete_day_df['DAY']>-10) &((first_complete_day_df['DAY']<31))].index

    #print('first_complete_day_df',first_complete_day_df)
    #print('final_cols_ to backfill',final_cols_)
    
    #print('pats_with_miss_vars',pats_with_miss_vars)
    
    #X_subset_=X_subset.copy()
    X_subset_=X_subset[X_subset['DAY']<=31].copy()

    #print('before backfill func',X_subset_.loc[X_subset_['USUBJID']=='TB-1022/53003',['DAY']+final_cols_].sort_values('DAY'))


    ## Loop over patients who have missing data in their early visits, and backward fill missing data
    #for pat in pats_with_miss_vars[:]:
    for pat,pat_df in X_subset_.groupby('USUBJID'):

        if pat not in pats_with_miss_vars:
            continue
        #print(pat)
        #pat_df=X_subset[X_subset['USUBJID']==pat]
            
    
        ## Get columns which have NaNs in the final columns containing input variables
        nan_cols_bool=pat_df.loc[:,['DAY']+final_cols_].sort_values('DAY').isna().any(axis=0)
        nan_cols=nan_cols_bool[nan_cols_bool.values].index.tolist()#+ ['dr_reg_study_drugs_cumul']
    
        ## backward fill those columns with the first observed value + insert backward filled data into the X_subset dataframe
        interp_df=pat_df[nan_cols].interpolate('bfill')
        X_subset_.loc[interp_df.index,nan_cols] = interp_df.values

    
    ## Get all the visits before DAY 5 of the study and drop those visits, where despite of the backward filling, there are still NaNs
    c=X_subset_[['DAY']+final_cols_].sort_values(by=['DAY']).groupby('USUBJID').apply(lambda x: x.loc[(x['DAY']<5),:]).dropna(how='any',subset=final_cols_,axis=0)#['USUBJID'].unique().shape

    #print(X_subset_[['DAY']+final_cols_].sort_values(by=['DAY']).groupby('USUBJID').apply(lambda x: x.loc[(x['DAY']<5),:]))
    
    ## If there are multiple early visits (usually in the week prior to therapy start, and then on the first 1-5 days), take the earliest timepoint as the baseline
    #. + drop drug regimen data columns, as theoretically no drugs have been taken yet
    #  + drop cumulate adverse events columns, as there was no therapy 
    X_subset_baseline=c.drop(columns=['USUBJID']).groupby('USUBJID',as_index=True).apply(lambda x: x.loc[(x.index[0]),:]).drop(columns=c.columns[c.columns.str.contains('dr_reg|drugs_cumul|cumul_toxgrade|CULTURE_STATUS')])#.dropna(how='any',subset=final_cols,axis=0)#['DAY']
    #X_subset_baseline['index']=np.nan
    
    return X_subset_baseline.reset_index()#.drop(columns='')


##=========================================
## SPLIT TRAINING AND TESTING DATA IN A STRATIFIED MANNER
#  - Standardise non-binary numerical columns with a standard scaler trained on the training dataset
#  - If training_data_type=='last_therapy_day', just keep the information from the last therapy day
                                                                                     
def create_std_training_testing_data(X,
                                     y,
                                     pat_ids_,
                                     test_size_ratio,
                                     rand_state,
                                     training_data_type,
                                     columns_to_drop,
                                     period_end_day,
                                     outcome_label,
                                     ther_arm_dur=None,
                                     cv_repeat_num=None,
                                     final_pat_ids_for_analysis=None):


    ## STRATIFY ON OUTCOME LABEL & STUDYID 
    ## IF STRATIFIED PATIENT IDS WERE ALREADY CALCULATED, LOAD THEM 
    if final_pat_ids_for_analysis is not None and cv_repeat_num is not None and ther_arm_dur is not None:

        X_train_pat_ids=final_pat_ids_for_analysis[period_end_day][cv_repeat_num][ther_arm_dur]['X_train_ids']
        X_test_pat_ids=final_pat_ids_for_analysis[period_end_day][cv_repeat_num][ther_arm_dur]['X_test_ids']
    
    ## IF STRATIFIED PATIENT IDS WERE NOT CALCULATED YET, PERFORM TRAIN-TEST SPLITTING
    else:   
        ##. ==> WITHIN STUDY ROC-AUC CALCULATION IS POSSIBLE, AS THERE ALWAYS WILL BE AT LEAST ONE UNFAVOUR. LABEL FROM BOTH STUDIES IN THE TEST SET
        pat_ids_=X['USUBJID'].unique().tolist()
        df_=y.loc[pat_ids_].reset_index()#
        df_['STUDYID']=df_['USUBJID'].str.split('/',expand=True)[0].values
        y_for_strat=df_[outcome_label].astype(str) + "_" + df_['STUDYID']#.astype(str)
        
        X_train_pat_ids, X_test_pat_ids, _, _ = train_test_split(pat_ids_, y_for_strat, test_size=test_size_ratio, stratify=y_for_strat,random_state=rand_state)

    

    #print('original num of patients',len(pat_ids_))
    #print('train-test split: train:',len(X_train_pat_ids),'test:',len(X_test_pat_ids))

    period_end_ind=period_end_days.index(period_end_day)

    if training_data_type=='last_therapy_day':
        periods_for_anal=period_end_days[period_end_ind:(period_end_ind+1)]

    if training_data_type=='all_days_in_period':
        periods_for_anal=period_end_days[:(period_end_ind+1)]

    
    x_train_l,x_test_l,full_l=[],[],[]

    for period_end_day_ in periods_for_anal:
        #print('period_end_day_',period_end_day_)
    
        if period_end_day_=='baseline':
            
            X_=backward_fill_and_extract_vars_at_baseline(X,columns_to_drop)
    
        if period_end_day_!='baseline':
            ## Drop columns that are not necessary for training 
            if len(columns_to_drop)>0:
                if 'DAY' in columns_to_drop:
                    columns_to_drop_wo_day=[x for x in columns_to_drop if x!='DAY']
                else:
                    columns_to_drop_wo_day=columns_to_drop
        
                X_=X.drop(columns=columns_to_drop_wo_day)   

        ## Ad the period name as a prefix for tha columns
        if training_data_type=='all_days_in_period':
            X_.columns=[f'{period_end_day_}_{col}' if col not in ['DAY','USUBJID','AGE','SEX','RACE'] else col for col in X_.columns ]

            ## Subset dataset to the period
            if isinstance(period_end_day_,int):
                X_=X_[(X_['DAY']<=period_end_day_)].copy()

            #print(X_['DAY'].max())
        
        ## Extract the last visit values in period
        #print(f'Num of patients at ==={period_end_day_}===:',X_['USUBJID'].unique().shape[0])

        X__ = X_.sort_values(by=['DAY']).groupby('USUBJID').apply(lambda x: x.loc[x.index[-1],:])

        ## Split last visit valies into training-testing data        
        X_train=X__.loc[X__.index.isin(X_train_pat_ids),:]
        X_test=X__.loc[X__.index.isin(X_test_pat_ids),:]
        X_full=X__.loc[X__.index.isin(pat_ids_),:]
        
        
        x_train_l.append(X_train)
        x_test_l.append(X_test)
        full_l.append(X_full)

    ## Concat dataeets of periods into one dataframe
    X_train=pd.concat(x_train_l,axis=1)
    X_test=pd.concat(x_test_l,axis=1)
    X_full=pd.concat(full_l,axis=1)

    #print('X_train shape after concatenation',X_train.shape)
   # print('X_test shape after concatenation',X_test.shape)
   

    ## Check number of visits with missing values per patient for each variable
    nan_visits_per_pat_per_var=(X_full.drop(columns=['USUBJID']).reset_index().groupby('USUBJID',as_index=False).apply(lambda x:x.isna().sum(axis=0)))

    ## Normalise by the number of patients, and select variables with a lower missing rate then a threshold to include in analysis
    #rel_missingness_of_vars=(nan_visits_per_pat_per_var.sum(axis=0).sort_values()/nan_visits_per_pat_per_var.shape[0])
    rel_missingness_of_vars=(nan_visits_per_pat_per_var.sum(axis=0)/nan_visits_per_pat_per_var.shape[0])
    non_sparse_vars = rel_missingness_of_vars[rel_missingness_of_vars<=0.05].index.tolist()

    #print(rel_missingness_of_vars[rel_missingness_of_vars<=0.05].sort_values())
    #print(non_sparse_vars)
    
    
    ## Drop rows (visits) with NaNs in the non-sparse-variables
    #X_train=X_train.dropna(how='any',axis=0)
    X_train=X_train[non_sparse_vars].dropna(how='any',axis=0)
    X_test=X_test[non_sparse_vars].dropna(how='any',axis=0)

    #print('X_train shape after dropping nan columns',X_train.shape)
    #print('X_test shape after dropping nan columns',X_test.shape)
    #print(np.sort(list(set(X_train.columns)))==np.sort(list(set(non_sparse_vars))))

    
    ## There will bb duplicated columns (USUBJID, DAY, ...) ==> drop them
    X_train = X_train.loc[:,~X_train.columns.duplicated()]
    X_test = X_test.loc[:,~X_test.columns.duplicated()]

    #print('after X_train.columns.duplicated()',X_train.shape)
    #print('after X_test.columns.duplicated()',X_test.shape)
    

    ## Some static variables have the same values at every period (i.e.m age, race, ...)
    ## ==> keep onlt the first instance of these columns
    static_cols_train=X_train.T.duplicated(keep='first')[X_train.T.duplicated(keep='first')].index.tolist()
    static_cols_test=X_test.T.duplicated(keep='first')[X_test.T.duplicated(keep='first')].index.tolist()

    ## Some drugs were stopped after a given period ==> their cumulative dose doesn't change in later periods, they get flagged as a static column
    #  ==> keep these drug regimen columns despite possibly having same values as their earlier period counterparts
    static_cols_train = [col for col in static_cols_train if 'dr_reg' not in col]
    static_cols_test = [col for col in static_cols_test if 'dr_reg' not in col]
    
    X_train=X_train.drop(columns=static_cols_train)
    X_test=X_test.drop(columns=static_cols_train)
    
    #print('static_cols_train to drop',static_cols_train)
    #print('X_train after dropping static_cols',X_train.shape)

    #X_train, X_test=scale_by_training_data(X_train, X_test)

    ## Subset y to training and testing set 
    # - If training_data_type=='last_therapy_day', this means one label per patient (i.e. Each patient has one row input)
    # - If training_data_type=='full', this means num_of_input_rows label per patient (i.e. Each patient has 'n' rows of input)
    #X_train_pat_ids=X_train['USUBJID'].values.tolist()
    #X_test_pat_ids=X_test['USUBJID'].values.tolist()
    X_train_pat_ids=X_train.index.tolist()
    X_test_pat_ids=X_test.index.tolist()

    #print(X_test)

    #print('X_train_pat_ids',X_train_pat_ids)
    #print('X_test_pat_ids',X_test_pat_ids)
    
    y_train=y.loc[X_train_pat_ids]
    y_test=y.loc[X_test_pat_ids]


    ## Drop 'DAY'  and USUBJID if still in columns
    if 'USUBJID' in X_train.columns:
        final_cols_to_drop=['DAY','USUBJID']
    else:
        final_cols_to_drop=['DAY']
    
    X_train=X_train.drop(columns=final_cols_to_drop)
    X_test=X_test.drop(columns=final_cols_to_drop)

    
    if 'ARM' in X_train.columns:
        X_train=X_train.drop(columns=['ARM'])
        X_test=X_test.drop(columns=['ARM'])



    
    ## Add STUDYID information to index for y_test and y_train
    y_test_=y_test.reset_index()
    #print('X_test_pat_ids',X_test_pat_ids)
    y_test_['STUDYID']=y_test_['USUBJID'].str.split('/',expand=True)[0].values
    y_test_=y_test_.set_index(['USUBJID','STUDYID'])
    
    y_train_=y_train.reset_index()
    y_train_['STUDYID']=y_train_['USUBJID'].str.split('/',expand=True)[0].values
    y_train_=y_train_.set_index(['USUBJID','STUDYID'])

    
    return X_train,X_test,y_train_,y_test_,X_train_pat_ids,X_test_pat_ids


###=========================================
### FUNCTION FOR TRAINING THE ML MODELS & CALCULATING CV ROC-AUC SCORES
def init_model(model_name,X_train,y_train_data,
                                k_folds,random_state,
                                outcome_label,
                                model_params,
                                weight_by_label_freq,
                                train_params) :

    from sklearn.model_selection import cross_val_score,RepeatedKFold,cross_validate,GridSearchCV
    from sklearn.linear_model import Lasso,LogisticRegressionCV,LogisticRegression
    from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier
    from xgboost import XGBClassifier
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier

    
    
    if weight_by_label_freq==True:
        target_label_freq=y_train_data[outcome_label].value_counts(normalize=True)
        
        target_label_freq=target_label_freq.sort_index(ascending=False)

        label_weights_dict={label:1/x for label,x in zip(target_label_freq.index.tolist(),
                                                         target_label_freq.values.tolist())}

        rf_label_weight='balanced' 
        gb_label_weight=y_train_data[outcome_label].map(label_weights_dict).values
        #print(gb_label_weight)
        #print('label_weights_dict',label_weights_dict)

    if weight_by_label_freq==False:

        if train_params['label_weights'] is not None:
            rf_label_weight={label:x for label,x in enumerate(train_params['label_weights'])}
            gb_label_weight=y_train_data[outcome_label].map(rf_label_weight).values
            label_weights_dict=rf_label_weight

            #print('label_weights_dict',rf_label_weight)

        if train_params['label_weights'] is None:
            rf_label_weight=None   
            gb_label_weight=None
            label_weights_dict=None
    
        

    if 'RandomForest' in model_name:

        label_weights=rf_label_weight
        label_weight_param_name='class_weight'
        
        '''
        rfCV=RandomForestClassifier(n_estimators=200, max_features='sqrt',random_state=random_state,
                                     class_weight=label_weights)
        params={'n_estimators':[300,500],'max_features':['log2', 'sqrt'],'max_depth':[3,5,7],}

        grid=GridSearchCV(rfCV,param_grid=params,cv=3,scoring='roc_auc',verbose=1,return_train_score=True)
        grid.fit(X_train,y_train_data)
        print(model_name,'parameter grid search...')

        '''
                                    
        model=RandomForestClassifier(class_weight=rf_label_weight,random_state=random_state,
                                     **model_params)
        
        #model=RandomForestClassifier(n_estimators=500, max_features='sqrt',random_state=random_state,
        #                             class_weight=label_weights)
        #model.fit(X_train, y_train_data)
        #feature_names=model.feature_names_in_
        #feature_importances=model.feature_importances_

        
    elif 'GradientBoost' in model_name:
        
        label_weights=gb_label_weight
        label_weight_param_name='sample_weight'
        '''
        gbCV=GradientBoostingClassifier(n_estimators=200, subsample=0.9,max_features='sqrt',random_state=random_state,\
                                        learning_rate=0.1)
        params={'n_estimators':[300,500],'max_features':['log2', 'sqrt'],'learning_rate':[0.1,0.3,0.5],}

        print(model_name,'parameter grid search...')                        
        grid=GridSearchCV(gbCV,param_grid=params,cv=3,scoring='roc_auc',verbose=1,return_train_score=True)
        grid.fit(X_train,y_train_data,sample_weight=label_weights)

        model=GradientBoostingClassifier(n_estimators=grid.best_params_['n_estimators'], subsample=0.9,
                                         max_features=grid.best_params_['max_features'],random_state=random_state,\
                                         learning_rate=grid.best_params_['learning_rate']) 
        '''                                                 
        
        model=GradientBoostingClassifier(random_state=random_state,**model_params)
        
        #model.fit(X_train, y_train_data,label_weights=gb_label_weight)
        #feature_names=model.feature_names_in_
        #feature_importances=model.feature_importances_

        
    elif 'XGBoost' in model_name:
        
        label_weights=gb_label_weight
        label_weight_param_name='sample_weight'
        
        '''
        xgbCV=XGBClassifier(n_estimators=200, max_depth=7, eta=0.1, subsample=1.0, colsample_bytree=0.8,\
                           random_state=random_state)
        params={'n_estimators':[250,500],'max_depth':[3,5,7],'eta':[0.1,0.3,0.5],}

        print(model_name,'parameter grid search...')                        
        grid=GridSearchCV(xgbCV,param_grid=params,cv=3,scoring='roc_auc',verbose=1,return_train_score=True)    
        grid.fit(X_train,y_train_data,sample_weight=label_weights)

        model=XGBClassifier(n_estimators=grid.best_params_['n_estimators'], max_depth=grid.best_params_['max_depth'],
                            eta=grid.best_params_['eta'], subsample=1.0, colsample_bytree=0.8,\
                            random_state=random_state)
        '''
        model=XGBClassifier(random_state=random_state,**model_params,colsample_bytree=0.8)
        #model.fit(X_train, y_train_data,sample_weight=label_weights)
        #feature_names=X_train.columns.tolist()
        #feature_importances=model.feature_importances_

        
    elif 'LogisticRegression' in model_name:
        label_weights=rf_label_weight
        label_weight_param_name='class_weight'
        
        model=LogisticRegression(penalty='elasticnet',solver='saga',class_weight=label_weights,max_iter=1000,**model_params)
        #model.fit(X_train, y_train_data)
        #feature_names=X_train.columns.tolist()
        #feature_importances=np.abs(model.coef_)

    
    elif 'SVC' in model_name:
        
        label_weights=rf_label_weight
        label_weight_param_name='class_weight'
        '''
        svc=SVC(kernel='rbf',gamma='scale', C=1.0,class_weight=rf_label_weight)
        params={'C':[1e-3,1e-2,1e-1,1e0,1e1,1e2]}
        #f1_macro_scorer=make_scorer(f1_score, average='macro')

        grid=GridSearchCV(svc,param_grid=params,cv=3,scoring='roc_auc',verbose=1,return_train_score=True)
        grid.fit(X_train,y_train_data)

        model=SVC(kernel='rbf',gamma='scale', C=grid.best_params_['C'],class_weight=rf_label_weight)
        '''
        model=SVC(gamma='scale',class_weight=rf_label_weight,**model_params)
        #model.fit(X_train, y_train_data)
        #feature_importances=np.abs(model.coef_)

        
    elif 'KNN' in model_name:
        
        label_weights=None
        label_weight_param_name=None
        '''
        knn=KNeighborsClassifier(n_neighbors=2)
        params={'n_neighbors':[2,5,10,25,50,100]}
        #f1_macro_scorer=make_scorer(f1_score, average='macro')

        grid=GridSearchCV(knn,param_grid=params,cv=3,scoring='roc_auc',verbose=1,return_train_score=True)
        grid.fit(X_train,y_train_data)
        model=KNeighborsClassifier(n_neighbors=grid.best_params_['n_neighbors'])
        '''
        model=KNeighborsClassifier(**model_params)
        #model.fit(X_train, y_train_data)
        #feature_importances=np.abs(model.coef_)   
        
    return model,label_weights,label_weights_dict




def init_survival_model(model_name, model_params, random_state,input_dim=None):
    
    from sksurv.ensemble import GradientBoostingSurvivalAnalysis
    from sksurv.linear_model import CoxnetSurvivalAnalysis
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBRegressor

    '''
    if model_name == 'GradientBoostingSurvival':
        return GradientBoostingSurvivalAnalysis(
            random_state=random_state,
            **model_params
        )

    elif model_name == 'CoxnetSurvival':
        return Pipeline([
            ('scaler', StandardScaler()),
            ('cox', CoxnetSurvivalAnalysis(
                fit_baseline_model=True,
                max_iter=1000,
                **model_params
            ))
        ])
    '''
    if 'GradientBoost' in model_name:
        model = GradientBoostingSurvivalAnalysis(**model_params,random_state=random_state)
        return model

    elif 'CoxnetSurvival' in model_name:
        model = CoxnetSurvivalAnalysis(fit_baseline_model=True,max_iter=10000,**model_params)
        return model

    elif 'XGBoost' in model_name:
        model = XGBRegressor(objective='survival:cox', 
                             eval_metric='cox-nloglik',
                             random_state=random_state,
                             **model_params)
        return model

    
    elif model_name in ['LogisticHazard', 'DeepHit', 'DeepHitSingle']:
        # return marker only; actual model is built in fit_pycox_model
        return None
        
    else:
    
        raise ValueError(f"Unknown survival model: {model_name}")

##=========================================
def compute_roc_auc(x, outcome_label):
    from sklearn.metrics import roc_auc_score
    """Computes the ROC-AUC score for binary or multiclass classification."""
    
    if isinstance(x,pd.DataFrame):
        y_true = x[outcome_label].values 
    if isinstance(x,pd.Series):    
        y_true = x.to_frame()[outcome_label].values  # True labels
    
    # Extract probability predictions for each class
    n_classes = len(np.unique(y_true))
    #print('x\n',x)
    y_score = x[[f'pred_class_{i}' for i in range(n_classes)]].values  # Extract correct class probabilities
    #print('np.sum(y_score,axis=1)',np.sum(y_score,axis=1))
    
    y_score /= y_score.sum(axis=1, keepdims=True)  
    
    if n_classes == 2:
        # **Binary Classification**
        return roc_auc_score(y_true, y_score[:, 1])  # Only use positive class
    else:
        # **Multiclass Classification**
        #per_class_auc = roc_auc_score(y_true, y_score, average=None, multi_class="ovr") 
        #print('Per class AUC:',per_class_auc)
        return roc_auc_score(y_true, y_score, average="macro", multi_class="ovo")  # Use OVO method



'''


##=========================================
def run_cv(X,y,k_folds,model_name,weight_by_label_freq,random_state,outcome_label,model_params,train_params,
          calibrate_model=False):
    
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import cross_validate
    pd.options.mode.chained_assignment = None
    
    
    pat_ids=y.index.get_level_values('USUBJID').unique()
    y_unique=y.reset_index().drop_duplicates(subset='USUBJID').set_index('USUBJID',drop=True)
    #print('y_unique',y_unique.loc[pat_ids,:])
   
    
    # Initialize a StratifiedKFold splitter
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=random_state)

    # Initialize lists to store the evaluation scores
    train_roc_auc_scores_all,test_roc_auc_scores_all=[],[]
    train_roc_auc_scores_per_study,test_roc_auc_scores_per_study=[],[]
    train_ids_,test_ids_,conf_metrics_list=[],[],[]
    
    cv_roc_auc_scores={}

    ## STRATIFY ON OUTCOME LABEL & STUDYID 
    ##. ==> WITHIN STUDY ROC-AUC CLAUCLATION IS POSSIBLE, AS THERE ALWAYS WILL BE AT LEAST ONE UNFAVOUR. LABEL FROM BOTH STUDIES IN THE TEST SET
    df_=y.reset_index().drop_duplicates(subset='USUBJID')#.set_index('USUBJID',drop=True)
    df_['STUDYID']=df_['USUBJID'].str.split('/',expand=True)[0].values
    df_=df_.set_index('USUBJID')
    y_for_strat=df_[outcome_label].astype(str) + "_" + df_['STUDYID']#.astype(str)

    pat_ids=df_.index
    n_of_classes=len(y[outcome_label].unique())

    cv_roc_auc_scores['inner_train_val_splits']={}
    
    # Loop through each fold
    #for train_index, test_index in skf.split(pat_ids, y_unique.loc[pat_ids,outcome_label]):
    for train_index, test_index in skf.split(pat_ids, y_for_strat.loc[pat_ids]):
        
        train_pat_ids=pat_ids[train_index]
        test_pat_ids=pat_ids[test_index]

        train_ids_.append(train_index)
        test_ids_.append(train_index)
        
        #print('train test index set',set(train_pat_ids)&set(test_pat_ids))
        
        train_mask=y.index.get_level_values('USUBJID').isin(train_pat_ids)
        test_mask=y.index.get_level_values('USUBJID').isin(test_pat_ids)
        
        
        # Split the data and standardise them by the training data's distribution
        #X_train_fold, X_test_fold = scale_by_training_data(X.loc[train_mask,:], X.loc[test_mask,:])
        X_train_fold, X_test_fold = X.loc[train_mask,:], X.loc[test_mask,:]
        y_train_fold,y_test_fold=y.loc[train_mask,:],y.loc[test_mask,:]

        
        
        ## Init model
        model,label_weights,label_weights_dict=init_model(model_name,X_train_fold,y_train_fold,
                                         k_folds,random_state,outcome_label,
                                         model_params,
                                         weight_by_label_freq,
                                         train_params) 
        
        ## Fit model
        if model_name in ['XGBoost','GradientBoost']:
            model.fit(X_train_fold, y_train_fold.values.ravel(),sample_weight=label_weights)
        else:
            model.fit(X_train_fold, y_train_fold.values.ravel())
                

        # Predict probabilities on the training and test data
        train_probabilities = model.predict_proba(X_train_fold)#[:, 1]
        test_probabilities = model.predict_proba(X_test_fold)#[:, 1]

        # Ensure probabilities are formatted correctly for multiclass
        if isinstance(train_probabilities, list):  
            train_probabilities = np.column_stack(train_probabilities)
            test_probabilities = np.column_stack(test_probabilities)

        # **Fix for 1D case**
        if train_probabilities.ndim == 1:
            train_probabilities = train_probabilities.reshape(-1, 1)
            test_probabilities = test_probabilities.reshape(-1, 1)

        #print('test_probabilities',test_probabilities.shape,test_probabilities)
        #print('y_train_fold',y_train_fold)
        
        y_train_fold_=y_train_fold.copy()

        # Store predictions in multiple columns (pred_class_0, pred_class_1, ...)
        for class_idx in range(train_probabilities.shape[1]):
            y_train_fold[f'pred_class_{class_idx}'] = train_probabilities[:, class_idx]
            y_test_fold[f'pred_class_{class_idx}'] = test_probabilities[:, class_idx]

               
        
        ## If patients come from multiple studies, calculate the ROC-AUC score within the studies as well
        if len(y.index.get_level_values('STUDYID').unique())>1:
            #y_train_fold['pred']=train_probabilities
            #y_test_fold['pred']=test_probabilities
            
            #train_roc_auc_per_study=y_train_fold.groupby('STUDYID').apply(lambda x:roc_auc_score(x[outcome_label],x['pred'],average="macro",multi_class='ovo'))
            #test_roc_auc_per_study=y_test_fold.groupby('STUDYID').apply(lambda x:roc_auc_score(x[outcome_label],x['pred'],average="macro",multi_class='ovo'))

            train_roc_auc_per_study = y_train_fold.groupby('STUDYID').apply(lambda x: compute_roc_auc(x, outcome_label))
            test_roc_auc_per_study = y_test_fold.groupby('STUDYID').apply(lambda x: compute_roc_auc(x, outcome_label))
            
            train_roc_auc_scores_per_study.append(train_roc_auc_per_study)
            test_roc_auc_scores_per_study.append(test_roc_auc_per_study)

        if calibrate_model==True:

            conf_metrics = calibrate_model_and_extract_confidence_metrics(
                                                               model=model,
                                                               X_train=X_train_fold,
                                                               X_test=X_test_fold,
                                                               y_train=y_train_fold_,
                                                               outcome_label=outcome_label,
                                                               label_weights_dict=label_weights_dict,
                                                               cv_roc_auc_scores=None,
                                                               cv_splitter=skf)
   
            
            ood_df = return_ood_metrics(X_train_fold,X_test_fold)
        
            #ood_conf_df=cv_roc_auc_scores['ood_conf']
            conf_metrics.loc[ood_df.index,ood_df.columns.tolist()] = ood_df.values
            conf_metrics_list.append(conf_metrics)

            
      
        # Calculate ROC AUC scores for all prediction across all studies
        #train_roc_auc=roc_auc_score(y_train_fold[outcome_label], train_probabilities)
        #test_roc_auc=roc_auc_score(y_test_fold[outcome_label], test_probabilities)

        #print('y_train_fold',y_train_fold)
        #print('y_test_fold',y_test_fold)

        #train_roc_auc = y_train_fold.apply(lambda x: compute_roc_auc(x, outcome_label))
        #test_roc_auc = y_test_fold.apply(lambda x: compute_roc_auc(x, outcome_label))
        train_roc_auc = compute_roc_auc(y_train_fold, outcome_label)
        test_roc_auc = compute_roc_auc(y_test_fold, outcome_label)

        #print('train_roc_auc',train_roc_auc)
        #print('test_roc_auc',test_roc_auc)

        # Append the scores to the lists
        train_roc_auc_scores_all.append(train_roc_auc)
        test_roc_auc_scores_all.append(test_roc_auc)
        
    ## Add train-test IDs
    cv_roc_auc_scores['inner_train_val_splits']['train_ids']=train_ids_
    cv_roc_auc_scores['inner_train_val_splits']['test_ids']=test_ids_

    if calibrate_model==True:
        ## Concatenate and add Confidence metrics + Mahalanobis and KNN confidence dataframes of the 5 validation sets
        cv_roc_auc_scores['inner_CV_conf_metrics'] = pd.concat(conf_metrics_list).loc[X.index,:]
    
    ## Add ROC-AUC values calculated across all studies to the result dictionary
    cv_roc_auc_scores['train_score']=train_roc_auc_scores_all
    cv_roc_auc_scores['test_score']=test_roc_auc_scores_all

    ## If multiple studies, join the per-study ROC-AUC scores into one dataframe containing scores per CV-fold 
    if len(y.index.get_level_values('STUDYID').unique())>1:
        
        train_roc_auc_per_study_df=pd.concat(train_roc_auc_scores_per_study,axis=1)
        test_roc_auc_per_study_df=pd.concat(test_roc_auc_scores_per_study,axis=1)
        
        cv_roc_auc_scores['train_score_per_study']=train_roc_auc_per_study_df
        cv_roc_auc_scores['test_score_per_study']=test_roc_auc_per_study_df
    
    return cv_roc_auc_scores 


'''
##=========================================
def is_pycox_model(model_name):
    return model_name in ['LogisticHazard', 'DeepHit', 'DeepHitSingle']


'''
##=========================================
def fit_pycox_model(model_name,
                    model_params,
                    random_state,
                    X_train,
                    durations_train,
                    events_train,
                    train_params=None,
                    X_val=None,
                    durations_val=None,
                    events_val=None):
    import numpy as np
    import torch
    import torchtuples as tt
    from torchtuples.practical import MLPVanilla
    from pycox.models import LogisticHazard, DeepHitSingle

    if train_params is None:
        train_params = {}

    torch.manual_seed(random_state)
    np.random.seed(random_state)

    X_train_np = np.asarray(X_train, dtype=np.float32)
    durations_train = np.asarray(durations_train, dtype=np.float32)
    events_train = np.asarray(events_train, dtype=np.int64)

    input_dim = X_train_np.shape[1]

    num_durations = model_params.get('num_durations', 20)
    hidden_nodes = model_params.get('hidden_nodes', [64, 32])
    dropout = model_params.get('dropout', 0.1)
    batch_norm = model_params.get('batch_norm', True)
    lr = model_params.get('lr', 1e-3)

    # 1) create label transform
    if model_name == 'LogisticHazard':
        labtrans = LogisticHazard.label_transform(num_durations)
    else:
        labtrans = DeepHitSingle.label_transform(num_durations)

    # 2) FIT label transform on the TRAINING FOLD
    y_train = labtrans.fit_transform(durations_train, events_train)

    # 3) NOW out_features is available
    net = MLPVanilla(
        in_features=input_dim,
        num_nodes=hidden_nodes,
        out_features=labtrans.out_features,
        batch_norm=batch_norm,
        dropout=dropout,
        output_bias=False
    )

    # 4) build model
    if model_name == 'LogisticHazard':
        model = LogisticHazard(net, tt.optim.Adam(lr=lr))
    else:
        alpha = model_params.get('alpha', 0.2)
        sigma = model_params.get('sigma', 0.1)
        model = DeepHitSingle(
            net,
            tt.optim.Adam(lr=lr),
            alpha=alpha,
            sigma=sigma
        )

    # validation data
    val_data = None
    if X_val is not None and durations_val is not None and events_val is not None:
        X_val_np = np.asarray(X_val, dtype=np.float32)
        durations_val = np.asarray(durations_val, dtype=np.float32)
        events_val = np.asarray(events_val, dtype=np.int64)
        y_val = labtrans.transform(durations_val, events_val)
        val_data = (X_val_np, y_val)

    batch_size = train_params.get('batch_size', 256)
    epochs = train_params.get('epochs', 100)
    verbose = train_params.get('verbose', False)

    model.fit(
        X_train_np,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        val_data=val_data
    )

    return {
        'model': model,
        'labtrans': labtrans
    }
'''
##=========================================
def fit_pycox_model(model_name,
                    model_params,
                    random_state,
                    X_train,
                    durations_train,
                    events_train,
                    train_params=None,
                    y_train_df=None,
                    X_val=None,
                    durations_val=None,
                    events_val=None):
    """
    Fit a pycox LogisticHazard or DeepHitSingle model with optional early stopping.

    Behavior
    --------
    1. If explicit validation data are provided, use them for early stopping.
    2. Otherwise, if early stopping is enabled, create an internal validation split
       from the provided training data.
    3. If early stopping is disabled, fit on all provided training data.

      If y_train_df is provided, the internal validation split is done:
    - at the patient level
    - stratified by RELAPSE + STUDYID

    Returns
    -------
    dict with keys:
        - 'model': fitted pycox model
        - 'labtrans': fitted label transform
        - 'train_idx': row indices used for fitting within provided training data
        - 'val_idx': row indices used for validation within provided training data
        - 'early_stopped': bool
        - 'n_trained_epochs': int
        - 'max_epochs': int
        - 'history_df': pandas DataFrame
    """
    import numpy as np
    import torch
    import torchtuples as tt

    from sklearn.model_selection import train_test_split
    from torchtuples.practical import MLPVanilla
    from pycox.models import LogisticHazard, DeepHitSingle

    if train_params is None:
        train_params = {}

    # ----------------------------
    # reproducibility
    # ----------------------------
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    # ----------------------------
    # inputs
    # ----------------------------
    X_all = np.asarray(X_train, dtype=np.float32)
    durations_all = np.asarray(durations_train, dtype=np.float64)
    events_all = np.asarray(events_train, dtype=np.int64)

    if X_all.ndim != 2:
        raise ValueError(f"X_train must be 2D, got shape {X_all.shape}")
    if len(X_all) != len(durations_all) or len(X_all) != len(events_all):
        raise ValueError("X_train, durations_train, and events_train must have same length")
    if np.isnan(durations_all).any() or np.isinf(durations_all).any():
        raise ValueError("durations_train contains NaN or inf")
    if (durations_all < 0).any():
        raise ValueError(
            "durations_train contains negative values. "
            "Pycox expects original non-negative durations, not XGBoost-style signed times."
        )
    if not np.isin(events_all, [0, 1]).all():
        raise ValueError("events_train must contain only 0/1 values")

    input_dim = X_all.shape[1]

    # ----------------------------
    # training hyperparameters
    # ----------------------------
    num_durations = model_params.get('num_durations', 6)
    hidden_nodes = model_params.get('hidden_nodes', [64, 32])
    dropout = model_params.get('dropout', 0.1)
    batch_norm = model_params.get('batch_norm', True)
    lr = model_params.get('lr', 1e-3)

    batch_size = model_params.get('batch_size', 256)
    epochs = model_params.get('epochs', 200)
  

    # Early stopping controls
    use_early_stopping = train_params.get('early_stopping', True)
    patience = train_params.get('patience', 10)
    min_delta = train_params.get('min_delta', 0.0)
    val_fraction = train_params.get('val_fraction', 0.2)

    verbose = train_params.get('verbose', False)

    # ----------------------------
    # validation split for early stopping
    # ----------------------------
    internal_train_idx = np.arange(len(X_all))
    internal_val_idx = None

    if X_val is not None and durations_val is not None and events_val is not None:
        # explicit validation data passed in
        X_fit = X_all
        durations_fit = durations_all
        events_fit = events_all

        X_val_np = np.asarray(X_val, dtype=np.float32)
        durations_val_np = np.asarray(durations_val, dtype=np.float64)
        events_val_np = np.asarray(events_val, dtype=np.int64)

        if np.isnan(durations_val_np).any() or np.isinf(durations_val_np).any():
            raise ValueError("durations_val contains NaN or inf")
        if (durations_val_np < 0).any():
            raise ValueError("durations_val contains negative values")
        if not np.isin(events_val_np, [0, 1]).all():
            raise ValueError("events_val must contain only 0/1 values")

    elif use_early_stopping and val_fraction > 0:
        if y_train_df is not None:
            # -----------------------------------------
            # PATIENT-LEVEL STRATIFIED SPLIT
            # -----------------------------------------
            y_df_ = y_train_df.copy()

            # STUDYID from USUBJID prefix
            y_df_['STUDYID'] = (
                y_df_.index.get_level_values('USUBJID')
                .str.split('/')
                .str[0]
            )

            # patient-level unique ids
            pat_ids = y_df_.index.get_level_values('USUBJID').unique()

            # patient-level relapse and study labels
            # assumes one label per patient across rows
            pat_df = (
                y_df_
                #.reset_index()
                .groupby('USUBJID')
                .agg({
                    'RELAPSE': 'first',
                    'STUDYID': 'first'
                })
            )

            y_for_strat = pat_df['RELAPSE'].astype(str) + "_" + pat_df['STUDYID'].astype(str)

            train_pat_ids, val_pat_ids = train_test_split(
                pat_ids,
                test_size=val_fraction,
                random_state=random_state,
                stratify=y_for_strat.loc[pat_ids]
            )

            train_mask = y_df_.index.get_level_values('USUBJID').isin(train_pat_ids)
            val_mask = y_df_.index.get_level_values('USUBJID').isin(val_pat_ids)

            internal_train_idx = np.where(train_mask)[0]
            internal_val_idx = np.where(val_mask)[0]

        else:
            # -----------------------------------------
            # FALLBACK: ROW-LEVEL STRATIFICATION
            # -----------------------------------------
            idx = np.arange(len(X_all))
            stratify = events_all if np.unique(events_all).size > 1 else None

            train_idx, val_idx = train_test_split(
                idx,
                test_size=val_fraction,
                random_state=random_state,
                stratify=stratify
            )

            internal_train_idx = train_idx
            internal_val_idx = val_idx

    

        X_fit = X_all[internal_train_idx]
        durations_fit = durations_all[internal_train_idx]
        events_fit = events_all[internal_train_idx]

        X_val_np = X_all[internal_val_idx]
        durations_val_np = durations_all[internal_val_idx]
        events_val_np = events_all[internal_val_idx]

    else:
        # no validation split, train on all data
        X_fit = X_all
        durations_fit = durations_all
        events_fit = events_all

        X_val_np = None
        durations_val_np = None
        events_val_np = None

    # ----------------------------
    # label transform
    # IMPORTANT: fit on fit/train subset only
    # ----------------------------
    if model_name == 'LogisticHazard':
        labtrans = LogisticHazard.label_transform(num_durations)
    elif model_name in ['DeepHit', 'DeepHitSingle']:
        labtrans = DeepHitSingle.label_transform(num_durations)
    else:
        raise ValueError(f"Unsupported pycox model_name: {model_name}")

    y_fit = labtrans.fit_transform(durations_fit, events_fit)

    # validation labels transformed with cuts learned on fit/train subset
    val_data = None
    if X_val_np is not None:
        y_val = labtrans.transform(durations_val_np, events_val_np)
        val_data = (X_val_np, y_val)

    # ----------------------------
    # network
    # ----------------------------
    net = MLPVanilla(
        in_features=input_dim,
        num_nodes=hidden_nodes,
        out_features=labtrans.out_features,
        batch_norm=batch_norm,
        dropout=dropout,
        output_bias=False
    )

    # ----------------------------
    # model
    # ----------------------------
    if model_name == 'LogisticHazard':
        model = LogisticHazard(net, tt.optim.Adam(lr=lr))
    else:
        alpha = model_params.get('alpha', 0.2)
        sigma = model_params.get('sigma', 0.1)
        model = DeepHitSingle(
            net,
            tt.optim.Adam(lr=lr),
            alpha=alpha,
            sigma=sigma
        )

    # ----------------------------
    # callbacks
    # torchtuples early stopping monitors validation loss by default
    # ----------------------------
    callbacks = None
    if use_early_stopping and val_data is not None:
        callbacks = [
            tt.callbacks.EarlyStopping(
                patience=patience,
                min_delta=min_delta
            )
        ]

    # ----------------------------
    # fit
    # ----------------------------
    log = model.fit(
        X_fit,
        y_fit,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        val_data=val_data,
        callbacks=callbacks
    )

    if verbose==True:
        n_trained_epochs = len(log.to_pandas())
        early_stopped = n_trained_epochs < epochs
        print(f"Trained for {n_trained_epochs}/{epochs} epochs")
        print(f"Early stopping happened: {early_stopped}")

    history_df = log.to_pandas()
    n_trained_epochs = len(history_df)
    early_stopped = n_trained_epochs < epochs

    return {
        'model': model,
        'labtrans': labtrans,
        'train_idx': internal_train_idx,
        'val_idx': internal_val_idx,
        'log': log,
        'history_df': history_df,
        'n_trained_epochs': n_trained_epochs,
        'max_epochs': epochs,
        'early_stopped': early_stopped,
    }




##=========================================
def pycox_surv_df_to_risk(surv_df):
    """
    Convert a predicted survival DataFrame into a scalar risk score.

    surv_df:
        rows = time grid
        columns = samples
    """
    import numpy as np

    times = surv_df.index.values.astype(float)
    surv = surv_df.values  # shape: (n_times, n_samples)

    if len(times) < 2:
        # fallback if only one time point exists
        expected_survival = surv[0, :]
    else:
        # trapezoidal integration of S(t) dt
        expected_survival = np.trapz(surv, times, axis=0)

    risk = -expected_survival
    return risk

##=========================================
def predict_pycox_risk(model_bundle, X):
    """
    Produce scalar risk scores for C-index calculation.
    """
    import numpy as np

    model = model_bundle['model']
    X_np = np.asarray(X, dtype=np.float32)
    surv_df = model.predict_surv_df(X_np)
    return pycox_surv_df_to_risk(surv_df)

##=========================================
def get_valid_ibs_times(y_train_df, y_test_df, n_times=100, time_col="RELAPSE_DAY"):
    from sksurv.metrics import integrated_brier_score
    
    train_times = y_train_df[time_col].to_numpy(dtype=float)
    test_times = y_test_df[time_col].to_numpy(dtype=float)

    # Must be inside test follow-up range
    lower = test_times.min()
    upper = test_times.max()

    # Also must not exceed training follow-up range for IPCW
    upper = min(upper, train_times.max())

    # upper must be strictly smaller than max follow-up
    upper = np.nextafter(upper, lower)

    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        return None

    times = np.linspace(lower, upper, n_times)
    return np.unique(times)



###=======================================
def calc_pycox_ibs_scores(model_bundle, X_train, y_train_df, X_eval, y_eval_df):
    """
    Calculate IBS and 1-IBS for a fitted pycox model on evaluation data.

    Returns
    -------
    ibs : float
    one_minus_ibs : float
    """
    import numpy as np
    from sksurv.util import Surv
    from sksurv.metrics import integrated_brier_score

    # structured arrays for censoring-adjusted IBS
    y_train_surv = Surv.from_arrays(
        event=y_train_df['RELAPSE'].astype(bool).values,
        time=y_train_df['RELAPSE_DAY'].astype(float).values
    )
    y_eval_surv = Surv.from_arrays(
        event=y_eval_df['RELAPSE'].astype(bool).values,
        time=y_eval_df['RELAPSE_DAY'].astype(float).values
    )


    '''
    pred_times = surv_df.index.values.astype(float)

    # IBS requires times within the valid follow-up range.
    # To stay safe, restrict to times strictly below the smallest of:
    # - max observed train time
    # - max observed eval time
    max_train_time = float(y_train_df['RELAPSE_DAY'].astype(float).max())
    max_eval_time = float(y_eval_df['RELAPSE_DAY'].astype(float).max())
    upper_time = min(max_train_time, max_eval_time)

    # strict upper bound for scikit-survival
    upper_time = np.nextafter(upper_time, -np.inf)

    times = pred_times[(pred_times > 0) & (pred_times <= upper_time)]
    times = np.unique(times)

    if len(times) < 2:
        return np.nan, np.nan
    '''

    def surv_df_to_array(surv_df, times):
        surv_times = surv_df.index.to_numpy(dtype=float)
        surv_mat = surv_df.to_numpy()  # shape: n_model_times x n_samples
    
        out = np.empty((surv_mat.shape[1], len(times)))
        for j in range(surv_mat.shape[1]):
            out[j, :] = np.interp(times, surv_times, surv_mat[:, j])
        return out

    
    times = get_valid_ibs_times(
        y_train_df=y_train_df,
        y_test_df=y_eval_df,
        n_times=100,
        time_col="RELAPSE_DAY"
    )
    if times is None or len(times) < 2:
        return np.nan, np.nan

     # pycox survival predictions: rows = times, cols = samples
    surv_df = model_bundle['model'].predict_surv_df(
        X_eval.values.astype('float32') if hasattr(X_eval, "values") else X_eval.astype('float32')
    )
    #surv_df = model_bundle["model"].predict_surv_df(X_eval.values)

    # estimate must be shape (n_samples, n_times)
    #surv_probs = surv_df.loc[times].T.values
    surv_probs = surv_df_to_array(surv_df, times)

    ibs = integrated_brier_score(
        survival_train=y_train_surv,
        survival_test=y_eval_surv,
        estimate=surv_probs,
        times=times
    )

    return ibs, 1.0 - ibs




##=========================================
def run_cv_survival(X,
                    y_df,          # dataframe with RELAPSE and RELAPSE_DAY columns
                    k_folds,
                    model_name,
                    random_state,
                    model_params,
                    train_params):
           

    from sklearn.model_selection import StratifiedKFold
    from sksurv.util import Surv
    from sksurv.metrics import concordance_index_censored
    import pandas as pd
    import numpy as np

    pd.options.mode.chained_assignment = None

    is_xgb = 'XGBoost' in model_name
    is_pycox = model_name in ['LogisticHazard', 'DeepHit', 'DeepHitSingle']


    # Build structured survival array — used for model fitting and scoring
    y_surv = Surv.from_arrays(
        event=y_df['RELAPSE'].astype(bool),
        time=y_df['RELAPSE_DAY'].astype(float)
    )

    # XGBoost format: censored = negative time, event = positive time
    y_xgb = y_df['RELAPSE_DAY'].astype(float).values.copy()
    y_xgb[y_df['RELAPSE'].astype(float).values == 0] *= -1
    # Ensure plain float32 — this is what causes "Other-9 is not supported"
    y_xgb = y_xgb.astype(np.float32)
    

    # Stratification label: event + study, same logic as your original
    y_df_ = y_df.copy()
    y_df_['STUDYID'] = y_df_.index.get_level_values('USUBJID').str.split('/').str[0]
    y_for_strat = y_df_['RELAPSE'].astype(str) + "_" + y_df_['STUDYID']

    pat_ids = y_df_.index.get_level_values('USUBJID').unique()

    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=random_state)


    train_c_index_all, test_c_index_all = [], []
    train_one_minus_ibs_all, test_one_minus_ibs_all = [], []
    train_ibs_all, test_ibs_all = [], []
    test_c_index_per_study = []

    for n_, (train_index, test_index) in enumerate(
            skf.split(pat_ids, y_for_strat.loc[pat_ids])):

        train_pat_ids = pat_ids[train_index]
        test_pat_ids = pat_ids[test_index]

        train_mask = y_df_.index.get_level_values('USUBJID').isin(train_pat_ids)
        test_mask = y_df_.index.get_level_values('USUBJID').isin(test_pat_ids)

        X_train_fold = X.loc[train_mask, :]
        X_test_fold = X.loc[test_mask, :]

        X_train_fold, X_test_fold=scale_by_training_data(X_train_fold, X_test_fold)

        y_test_events  = y_df_.loc[test_mask,  'RELAPSE'].astype(bool).values
        y_test_times   = y_df_.loc[test_mask,  'RELAPSE_DAY'].astype(float).values
        y_train_events = y_df_.loc[train_mask, 'RELAPSE'].astype(bool).values
        y_train_times  = y_df_.loc[train_mask, 'RELAPSE_DAY'].astype(float).values
        y_train_df  = y_df_.loc[train_mask, :]
        y_test_df  = y_df_.loc[test_mask, :]


        # Build model
        if is_pycox:
            model_bundle = fit_pycox_model(
                                model_name=model_name,
                                model_params=model_params,
                                random_state=random_state,
                                X_train=X_train_fold.values,
                                y_train_df=y_train_df,
                                durations_train=y_train_times,
                                events_train=y_train_events.astype(int),
                                train_params=train_params
                            )
                                    

            train_risk = predict_pycox_risk(model_bundle, X_train_fold.values)
            test_risk = predict_pycox_risk(model_bundle, X_test_fold.values)

            # IBS / 1-IBS
            train_ibs, train_one_minus_ibs = calc_pycox_ibs_scores(
                model_bundle=model_bundle,
                X_train=X_train_fold,
                y_train_df=y_train_df,
                X_eval=X_train_fold,
                y_eval_df=y_train_df
            )
            test_ibs, test_one_minus_ibs = calc_pycox_ibs_scores(
                model_bundle=model_bundle,
                X_train=X_train_fold,
                y_train_df=y_train_df,
                X_eval=X_test_fold,
                y_eval_df=y_test_df
            )

            train_ibs_all.append(train_ibs)
            test_ibs_all.append(test_ibs)
            train_one_minus_ibs_all.append(train_one_minus_ibs)
            test_one_minus_ibs_all.append(test_one_minus_ibs)
        
        else:
            model = init_survival_model(model_name, model_params, random_state)

            # Key fix: branch on model type for both fit and label format
            if is_xgb:
                model.fit(
                    X_train_fold,
                    y_xgb[train_mask],  # plain float32 array
                    verbose=False
                )
            else:
                model.fit(X_train_fold, y_surv[train_mask])

            train_risk = model.predict(X_train_fold)
            test_risk  = model.predict(X_test_fold)

        if y_test_events.sum() == 0 or (~y_test_events).sum() == 0:
            continue

        # Store predictions back onto the test dataframe
        y_test_df = y_test_df.copy()
        y_test_df['pred_risk'] = test_risk

        # Overall C-index
        train_c = concordance_index_censored(
            y_train_events, y_train_times, train_risk
        )[0]
        test_c = concordance_index_censored(
            y_test_events, y_test_times, test_risk
        )[0]

        train_c_index_all.append(train_c)
        test_c_index_all.append(test_c)

        # Per-study C-index — same logic as your per-study AUC
        if y_df_['STUDYID'].nunique() > 1:
            def c_index_for_group(grp):
                events = grp['RELAPSE'].astype(bool).values
                times = grp['RELAPSE_DAY'].astype(float).values
                risks = grp['pred_risk'].values
                # Need at least one event and one non-event to compute C-index
                if events.sum() == 0 or (~events).sum() == 0:
                    return np.nan
                return concordance_index_censored(events, times, risks)[0]

            #print('y_test_df',y_test_df)
            test_c_per_study  = (y_test_df
                                .drop(columns=['STUDYID']) # flatten any MultiIndex
                                .reset_index()                                
                                .groupby('STUDYID')
                                .apply(c_index_for_group))
            test_c_index_per_study.append(test_c_per_study)

    #results = {
    #    'train_score': train_c_index_all,
    #    'test_score': test_c_index_all,
    #}

    # - for pycox models: optimize 1-IBS
    # - for others: optimize C-index
    if is_pycox:
        results = {
            'train_score': train_one_minus_ibs_all,
            'test_score': test_one_minus_ibs_all,
            #'train_1_minus_ibs': train_one_minus_ibs_all,
            #'test_1_minus_ibs': test_one_minus_ibs_all,
            #'train_ibs': train_ibs_all,
            #'test_ibs': test_ibs_all,
            'train_c_index': train_c_index_all,
            'test_c_index': test_c_index_all,
        }
    else:
        results = {
            'train_score': train_c_index_all,
            'test_score': test_c_index_all,
            #'train_c_index': train_c_index_all,
            #'test_c_index': test_c_index_all,
        }

    if test_c_index_per_study:
        results['test_score_per_study'] = pd.concat(
            test_c_index_per_study, axis=1
        )

    return results

def make_xgb_survival_labels(y_df):
    labels = y_df.copy()#['RELAPSE_DAY'].astype(float).copy()
    censored_mask = y_df['RELAPSE'].astype(float) == 0
    labels.loc[censored_mask,'RELAPSE_DAY'] = -1 * labels.loc[censored_mask,'RELAPSE_DAY'].values
    return labels



##=========================================
## RUN CV OF GIVEN MODEL & TRAIN FINAL MODEL AFTERWARDS
def run_parameter_search(model_name,
                         X_train,
                         y_train_data,
                         k_folds,
                         random_state,
                         outcome_label,
                         param_search_dict,
                         #weight_by_label_freq,
                         train_params,
                         #calibrate_model,
                         #survival_anal,
                        ):


    ## For XGBoost, set the right-censored datapoint to negative, as is convention
    #if 'XGBoost' in model_name:
    #    y_train_data = make_xgb_survival_labels(y_df=y_train_data)
    
    from tqdm.auto import tqdm
    
    param_search_results={}
        
    ### RUN CV WITH DIFFERENT PARAMETER SETS & SAVE THE CV-RESULTS INTO A DICT
    parameter_combinations = list(itertools.product(*param_search_dict[model_name].values()))
    parameter_sets=[{key: value for key, value in zip(param_search_dict[model_name].keys(), combination)} for combination in parameter_combinations]

    #for n, model_params in enumerate(parameter_sets):
    for n, model_params in enumerate(tqdm(parameter_sets[:], desc="Processing", unit="parameter_set")):

        ## Create string of the parameters
        pairs=[f"{key}:{value}" for key, value in model_params.items()]
        param_set_string='-'.join(pairs)

        #print('model_params',model_params)
  
        #print(f'Running CV of {model_name} model with parameters:{param_set_string}')
        random_state_=random_state #+n
              
        ## CALCULATE CV-SCORES
        '''
        cv_roc_auc_scores=run_cv(X_train,
                                 y_train_data,
                                 k_folds,
                                 model_name,
                                 weight_by_label_freq,
                                 random_state_,outcome_label,
                                 model_params,
                                train_params,
                                 survival_anal
                                )
        '''

        results = run_cv_survival(X=X_train,
                                y_df=y_train_data,          # dataframe with RELAPSE and RELAPSE_DAY columns
                                k_folds=k_folds,
                                model_name=model_name,
                                random_state=random_state_,
                                model_params=model_params,
                                train_params=train_params)
        ## SAVE RESULT OF CV WITH GIVEN PARAMETER SET
        param_search_results[param_set_string]=results

        
    return param_search_results



##=========================================
def extract_best_model_params(param_search_results,metric_func,num_of_top_models_per_cv,
                              average_models_across_splits=False):
    
    def is_float(s):
        try:
            a=float(s)
            return True
        except ValueError:
            return False
    
    
    ## Init dictionary to hold the best results for each CV-repetition
    best_cv_rep_results={}
    
    ## Loop through all thr CV-repeats and extract the CV-scores of the validation splits
    for cv_rep_num in param_search_results['param_search_results'].keys():
        
        cv_rep_param_search_results = param_search_results['param_search_results'][cv_rep_num]

        ## Construct a dataframe with the parmeter sets and the corresponding mean or median of the validation ROC-AUC of model trained with parameter set
        param_roc_auc_df=pd.DataFrame.from_dict({param_set_string:cv_rep_param_search_results[param_set_string]['test_score'] for param_set_string in cv_rep_param_search_results.keys()})
        best_model_params = np.mean(param_roc_auc_df,axis=0).sort_values(ascending=False).index[:num_of_top_models_per_cv]

        best_params_dict={}
        for n,best_param_set in enumerate(best_model_params):
            
            ## Convert the best parameter-set to a dictionary
            best_model_params={key_val_pair.split(':')[0]:key_val_pair.split(':')[1] for key_val_pair in best_param_set.split('-')}
            best_params_dict[n]=best_model_params
            #print('best_model_params',best_model_params)
    
        
        #best_model_params={key:float(value) if is_float(value) else value for key,value in best_model_params.items() }
        #print('best_model_params',best_model_params)
        
        best_cv_rep_results[cv_rep_num]=best_params_dict
        
        #print(f'Best CV-score {best_score} in CV-rep {cv_rep_num}, best params: {best_model_params}')
    
    ## AVERAGE THE PARAMETERS OF THE BEST MODELS ACROSS ALL SPLITS
    if average_models_across_splits==True:
        raise ValueError('Not correctly implemented!')

        
        ## Initialize dictionary for averaging the parameters of the best parameter-sets/CV-repeat
        best_model_params_avg={}
        
        ## Average the parameters values across the different CV-splits, and save them into a dictionary
        n = len(best_cv_rep_results)
        for param_name in best_cv_rep_results[[*best_cv_rep_results][0]].keys():
            
            if is_float(best_cv_rep_results[[*best_cv_rep_results][0]][param_name])==True:
            
                best_model_params_avg[param_name] = sum(best_cv_rep_results[cv_rep_num][param_name] for cv_rep_num in best_cv_rep_results.keys()) / n
    
                if best_model_params_avg[param_name]>1:
                    best_model_params_avg[param_name]=int(best_model_params_avg[param_name])
                    
            if is_float(best_cv_rep_results[[*best_cv_rep_results][0]][param_name])==False:
                best_model_params_avg[param_name]= list(set(best_cv_rep_results[cv_rep_num][param_name] for cv_rep_num in best_cv_rep_results.keys()))[0]
        
        return best_model_params_avg
    
    ## TAKE THE PARAMETERS OF THE BEST MODELOF GIVEN SPLIT
    if average_models_across_splits==False:

        for cv_rep_num in best_cv_rep_results.keys():

            for n in range(len(best_cv_rep_results[cv_rep_num].keys())):
                
                best_model_params=best_cv_rep_results[cv_rep_num][n]
           

                ## If parameter is float and larger than 1, convert it to integer, as scklearn will throw an error
                #for param_name in best_cv_rep_results[[*best_cv_rep_results][0]].keys():
                for param_name in best_model_params.keys():

                    #print(best_model_params[param_name],type(best_model_params[param_name]),'is float',is_float(best_model_params[param_name]))
                    if is_float(best_model_params[param_name])==True and float(best_model_params[param_name])>1:
                        best_model_params[param_name]=int(best_model_params[param_name])
                    
                    if is_float(best_model_params[param_name])==True and float(best_model_params[param_name])<=1:
                        best_model_params[param_name]=float(best_model_params[param_name])

                best_cv_rep_results[cv_rep_num][n]=best_model_params
        
        return best_cv_rep_results

'''
##=========================================
## RUN CV OF GIVEN MODEL & TRAIN FINAL MODEL AFTERWARDS
def calc_roc_auc_score_of_model(model_name,
                                X_train,
                                y_train_data,
                                k_folds,
                                random_state,
                                outcome_label,
                                model_params,
                                weight_by_label_freq,
                               train_params,
                               calibrate_model):
        
    ### RUN CV & TRAIN MODEL AFTERWARDS
    if model_name in ['XGBoost','GradientBoost']:
        
        ## CALCULATE CV-SCORES
        cv_roc_auc_scores=run_cv(X_train,y_train_data,k_folds,
                                 model_name,weight_by_label_freq,
                                 random_state,outcome_label,
                                 model_params,
                                 train_params,
                                calibrate_model)
        
        ## INITIALIZE MODEL & TRAIN 
        model,label_weights,label_weights_dict=init_model(model_name,X_train,y_train_data,
                                                          k_folds,random_state,outcome_label,
                                                          model_params,
                                                          weight_by_label_freq,
                                                          train_params) 
        
        #X_train,_ = scale_by_training_data(X_train, X_train)
        model.fit(X_train, y_train_data,sample_weight=label_weights)
        
    else:
        
        ## CALCULATE CV-SCORES
        cv_roc_auc_scores=run_cv(X_train,y_train_data,k_folds,
                                 model_name,weight_by_label_freq,
                                 random_state,outcome_label,
                                 model_params,
                                 train_params,
                                calibrate_model)
        
        ## INITIALIZE MODEL & TRAIN 
        model,label_weights,label_weights_dict=init_model(model_name,X_train,y_train_data,
                                                          k_folds,random_state,outcome_label,
                                                          model_params,
                                                          weight_by_label_freq,
                                                          train_params) 

        #X_train,_ = scale_by_training_data(X_train, X_train)
        model.fit(X_train, y_train_data)
        
    return model,cv_roc_auc_scores,label_weights_dict
'''
##=========================================
## RUN CV OF GIVEN MODEL & TRAIN FINAL MODEL AFTERWARDS
'''
##=========================================
## RUN CV OF GIVEN MODEL & TRAIN FINAL MODEL AFTERWARDS
def calc_roc_auc_score_of_model(model_name,
                                X_train,
                                y_train_data,
                                k_folds,
                                random_state,
                                outcome_label,
                                model_params,
                                weight_by_label_freq,
                                train_params,
                                calibrate_model):
        
    ### RUN CV & TRAIN MODEL AFTERWARDS
    if model_name in ['XGBoost','GradientBoost']:
        
        ## CALCULATE CV-SCORES
        cv_roc_auc_scores=run_cv(X_train,y_train_data,k_folds,
                                 model_name,weight_by_label_freq,
                                 random_state,outcome_label,
                                 model_params,
                                 train_params,
                                 calibrate_model)
        
        ## INITIALIZE MODEL & TRAIN 
        model,label_weights,label_weights_dict=init_model(model_name,X_train,y_train_data,
                                                          k_folds,random_state,outcome_label,
                                                          model_params,
                                                          weight_by_label_freq,
                                                          train_params) 
        
        #X_train,_ = scale_by_training_data(X_train, X_train)
        model.fit(X_train, y_train_data,sample_weight=label_weights)
        
    else:
        
        ## CALCULATE CV-SCORES
        cv_roc_auc_scores=run_cv(X_train,y_train_data,k_folds,
                                 model_name,weight_by_label_freq,
                                 random_state,outcome_label,
                                 model_params,
                                 train_params,
                                 calibrate_model)
        
        ## INITIALIZE MODEL & TRAIN 
        model,label_weights,label_weights_dict=init_model(model_name,X_train,y_train_data,
                                                          k_folds,random_state,outcome_label,
                                                          model_params,
                                                          weight_by_label_freq,
                                                          train_params) 

        #X_train,_ = scale_by_training_data(X_train, X_train)
        model.fit(X_train, y_train_data)
        
    return model,cv_roc_auc_scores,label_weights_dict

'''

##=========================================
## RUN CV OF GIVEN MODEL & TRAIN FINAL MODEL AFTERWARDS
def calc_c_index_score_of_model(model_name,
                                X_train,
                                y_train_data,
                                k_folds,
                                random_state,
                                outcome_label,
                                model_params,
                                #weight_by_label_freq,
                                train_params,
                                #calibrate_model
                               ):
    from sksurv.util import Surv

    
    is_xgb = 'XGBoost' in model_name
    is_pycox = model_name in ['LogisticHazard', 'DeepHit', 'DeepHitSingle']


    ## CALCULATE C-INDEX -SCORES 
    cv_c_index_scores = run_cv_survival(X = X_train,
                                        y_df=y_train_data,          # dataframe with RELAPSE and RELAPSE_DAY columns
                                        k_folds=k_folds,
                                        model_name=model_name,
                                        random_state=random_state,
                                        model_params=model_params,
                                        train_params=train_params)
        
    ### RUN CV & TRAIN MODEL AFTERWARDS

    if is_pycox:
        durations_train = y_train_data['RELAPSE_DAY'].astype(float).values
        events_train = y_train_data['RELAPSE'].astype(int).values

        model = fit_pycox_model(
            model_name=model_name,
            model_params=model_params,
            random_state=random_state,
            X_train=X_train.values if hasattr(X_train, "values") else X_train,
            durations_train=durations_train,
            events_train=events_train,
            train_params=train_params
        )
         
    elif is_xgb:
        

        # XGBoost format: censored = negative time, event = positive time
        y_xgb = y_train_data['RELAPSE_DAY'].astype(float).values.copy()
        y_xgb[y_train_data['RELAPSE'].astype(float).values == 0] *= -1
        y_xgb = y_xgb.astype(np.float32)

        ## INITIALIZE MODEL & TRAIN 
        model = init_survival_model(model_name, model_params, random_state)
        model.fit(X_train,y_xgb, verbose=False)

    else:

        ## INITIALIZE MODEL & TRAIN 
        model = init_survival_model(model_name, model_params, random_state)

        # Build structured survival array — used for model fitting and scoring
        y_surv = Surv.from_arrays(event=y_train_data['RELAPSE'].astype(bool),
                                  time=y_train_data['RELAPSE_DAY'].astype(float))
  
        model.fit(X_train, y_surv)

        
    return model,cv_c_index_scores


##=========================================
## Setup function for calculating elapsed time
def print_elapsed_time(start,stop):
    # Calculate the elapsed time in seconds
    elapsed_seconds = stop - start
    
    # Convert elapsed time to hours and minutes
    elapsed_minutes, elapsed_seconds = divmod(int(elapsed_seconds), 60)
    elapsed_hours, elapsed_minutes = divmod(elapsed_minutes, 60)
    
    # Print the result in the desired format
    print(f"Elapsed time:{elapsed_hours} hours:{elapsed_minutes} minutes")
    
    
    
##=========================================  
'''
def draw_pca_biplot(score,coeff,y,loading_rel_length_thr,legend_title,labels=None):
    from adjustText import adjust_text
    
    fig,ax=plt.subplots(1,1,figsize=(15,10))
    xs = score[:,0]
    ys = score[:,1]
    
    
    load_vect_lengths=np.sqrt(coeff[:,0]**2 + coeff[:, 1]**2)
    norm_load_vect_lengths=load_vect_lengths/np.max(load_vect_lengths)

    coeff_filt=coeff[norm_load_vect_lengths>loading_rel_length_thr]
    labels=[x.split('_STD_NUM_RESULT')[0] for x in labels]
    labels=[x.split('dr_reg_')[-1] for x in labels]
    labels_filt=np.array(labels)[norm_load_vect_lengths>loading_rel_length_thr]
    
    n = coeff_filt.shape[0]
    
    scalex = 1.0/(xs.max() - xs.min())
    scaley = 1.0/(ys.max() - ys.min())
    scatter=ax.scatter(xs * scalex,ys * scaley, c = y,s=4)
    
    # produce a legend with the unique colors from the scatter
    h=scatter.legend_elements()[0]
    l=[id2label[int(x.split('{')[-1].split('}')[0])] for x in scatter.legend_elements()[1]]
    legend1 = ax.legend(handles=h,labels=l,loc="best", title=legend_title)
    ax.add_artist(legend1)
    
    for i in range(n):
        ax.arrow(0, 0, coeff_filt[i,0], coeff_filt[i,1],color = 'r',alpha = 0.5)
        if labels is None:
            ax.text(coeff_filt[i,0]* 1.15, coeff_filt[i,1] * 1.15, "Var"+str(i+1), color = 'g', ha = 'center', va = 'center')
        else:
            ax.text(coeff_filt[i,0]* 1.2, coeff_filt[i,1] * 1.2, labels_filt[i], color = 'g', ha = 'center', va = 'center')
    #plt.xlim(-1,1)
    #plt.ylim(-1,1)
    ax.set_xlabel("PC{}".format(1))
    ax.set_ylabel("PC{}".format(2))
    plt.grid()   
    
 '''   
##=========================================  
def draw_pca_biplot(score, coeff, y, loading_rel_length_thr, legend_title, labels=None):
    from adjustText import adjust_text
    
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    xs = score[:, 0]
    ys = score[:, 1]
    
    load_vect_lengths = np.sqrt(coeff[:, 0]**2 + coeff[:, 1]**2)
    norm_load_vect_lengths = load_vect_lengths / np.max(load_vect_lengths)

    coeff_filt = coeff[norm_load_vect_lengths > loading_rel_length_thr]
    labels = [x.split('_STD_NUM_RESULT')[0] for x in labels]
    labels = [x.split('dr_reg_')[-1] for x in labels]
    labels_filt = np.array(labels)[norm_load_vect_lengths > loading_rel_length_thr]
    
    n = coeff_filt.shape[0]
    
    scalex = 1.0 / (xs.max() - xs.min())
    scaley = 1.0 / (ys.max() - ys.min())
    scatter = ax.scatter(xs * scalex, ys * scaley, c=y, s=4)
    
    # produce a legend with the unique colors from the scatter
    h = scatter.legend_elements()[0]
    l = [id2label[int(x.split('{')[-1].split('}')[0])] for x in scatter.legend_elements()[1]]
    legend1 = ax.legend(handles=h, labels=l, loc="best", title=legend_title)
    ax.add_artist(legend1)
    
    for i in range(n):
        ax.arrow(0, 0, coeff_filt[i, 0], coeff_filt[i, 1], color='r', alpha=0.5)
    
    texts = []
    for i in range(n):
        if labels is None:
            texts.append(ax.text(coeff_filt[i, 0] * 1.15, coeff_filt[i, 1] * 1.15, "Var"+str(i+1), color='g', ha='center', va='center'))
        else:
            texts.append(ax.text(coeff_filt[i, 0] * 1.15, coeff_filt[i, 1] * 1.15, labels_filt[i], color='g', ha='center', va='center'))
    
    adjust_text(texts, arrowprops=dict(arrowstyle="->", color='b', lw=0.5))
    
    ax.set_xlabel("PC{}".format(1))
    ax.set_ylabel("PC{}".format(2))
    plt.grid()   



##=========================================  
## Select variables from temporal-type data types, where enough patients (==num_of_pats * temp_col_threshold) have entries for that variable
#  - Temporal variables: 
#  (ae: adverse events,ce: clinical events, mh: medical history, cm: concomitant medication, cmind: indication for concomitant medication)
def select_temporal_cols_with_suff_pat_data(temp_data_name,X_subset,temp_col_threshold): 
    variables_per_patient_all=pd.read_csv('../data/all_pat_variables_with_reliable_therapy_data.csv.gz',index_col=0)
    
    vars_per_pat=variables_per_patient_all.loc[X_subset['USUBJID'].unique(),:]
    a=vars_per_pat.loc[:,vars_per_pat.columns.str.startswith(temp_data_name)].sum()
    temp_cols_with_suff_data=a[a>len(X_subset['USUBJID'].unique())*temp_col_threshold].index.tolist()

    return temp_cols_with_suff_data


##=========================================  
## For patients, who stopped therapy earlier than the scheduled duration of the study, the cumulative drug doses are set to 0 for those days, 
#. where the drugs weren't taken anymore. This originates from the way the drug regimen was extracted in step s4. 
#  To remedy this problem, forward fill the last cumulative dose for those days.
def ffill_dr_reg_cumul_cols(X_subset):
    
    ## Extract dr_reg cumulative columns + DAY and USUBJID
    dr_reg_ffill_cols=['DAY','USUBJID']+X_subset.columns[X_subset.columns.str.startswith('dr_reg')].tolist()

    ## 1. Replace the 0s with NaNs==> first therapy day & days where wasn't taken anymore are becoming NaNs
    ## 2. Forward fill ==> only the days without drug threapy get filled with last cumulative dose
    ## 3. Fill NaNs with 0==> fill the first day of therapy with a 0, indicating no drugs were taken yet
    dr_reg_ffill=X_subset.loc[:,dr_reg_ffill_cols].groupby('USUBJID',as_index=False).apply(lambda x: x.sort_values(by='DAY').replace(0, np.nan).ffill().fillna(0))

    ## Merge the original data with the ffilled drug regimen data
    X_subset_ffill=pd.merge(X_subset.loc[:,~X_subset.columns.str.contains('dr_reg')],\
                            dr_reg_ffill.loc[:,dr_reg_ffill_cols],on=['DAY','USUBJID'],how='outer')

    return X_subset_ffill




##========================================= 
### 1. COLLECT PATIENTS, WHO ONLY HAVE FAVOURABLE OUTCOMES AT END OF TREATMENT & AT ALL FOLLOW-UP TIMEPOINTS 
    #. ==>TB-1021: 12 & 18 MONTHS, TB-1022: 18 & 24 MONTHS

### 2. COLLECT PATIENTS, WHO HAVE FAVOURABLE OUTCOME AT END OF TREATMENT, BUT HAVE AT LEAST ONE UNFAVOURABLE OUTCOME AT ANY OF THE FOLLOW-UP TIMEPOINTS 
    #. ==>TB-1021: 12 & 18 MONTHS, TB-1022: 18 & 24 MONTHS

def extract_rifaquin_relapse():
    
    data=load_merged_data_of_lab_vars()
    arm_df=data.drop_duplicates('USUBJID')[['USUBJID','ARM']].set_index('USUBJID')
    del data
    
    
    de=pd.read_csv('../../C-Path_data/preprocessing/disposition_events.csv',low_memory=True)
    de = de.set_index('USUBJID')


    
    outcome_tb1020 = pd.read_csv('../data/tb_1020_outcome.csv.gz')
    tb_1020_pat_df = outcome_tb1020[outcome_tb1020['UNFAVOURABLE_OUTCOME_CATEGORY_AT_18_MONTHS'].isin(['FAVOURABLE','RELAPSE'])]
    tb_1020_pat_df = tb_1020_pat_df.rename(columns={'Unnamed: 0':'USUBJID',})

    #df_tb_20 = data[data['USUBJID'].isin(tb_1020_pat_df['USUBJID'].tolist())]



    tb_1020_pat_df = tb_1020_pat_df.set_index('USUBJID')
    tb_1020_pat_df['last_therapy_day'] = de.loc[tb_1020_pat_df.index,'COMPLETION CONTINUATION PHASE'].values
    
    tb_1020_pat_df['RELAPSE']=tb_1020_pat_df['UNFAVOURABLE_OUTCOME_CATEGORY_AT_18_MONTHS'].replace({'FAVOURABLE':0,'RELAPSE':1})
    tb_1020_pat_df['RELAPSE_DAY']= tb_1020_pat_df['TIME_TO_EVENT'].values
    
    
    tb_1020_pat_df['DAYS_BETWEEEN_THERAPY_END_AND_RELAPSE'] = (tb_1020_pat_df['RELAPSE_DAY'] - tb_1020_pat_df['last_therapy_day']).values
    tb_1020_pat_df.loc[tb_1020_pat_df['RELAPSE']==0,['DAYS_BETWEEEN_THERAPY_END_AND_RELAPSE','RELAPSE_DAY']]=np.nan

    #df_tb_20 = data[data['USUBJID'].isin(tb_1020_pat_df.reset_index()['USUBJID'].tolist())]
   
    tb_1020_pat_df['ARM'] = arm_df.loc[tb_1020_pat_df.reset_index()['USUBJID'],'ARM'].values
    tb_1020_pat_df['STUDYID'] = 'Rifaquin'

    
    return tb_1020_pat_df




def extract_21_22_relapse_pats(include_rifaquin=False):

    print('Extracting relapse information...')
    
    ### ====== LOAD ALL PATIENTS DATA =======
    fn='../data/tb21_22_2984_pats_22_vars_result_at_end_of_treatment_preproc_data_with_imp.csv.gz'
    X_subset=pd.read_csv(fn,index_col=0)
    X_subset=X_subset.rename(columns=lambda x: x.replace('<', 'lower than'))
    X_subset=X_subset.rename(columns=lambda x: x.replace('>', 'higher than'))
    
    
    #### ================================ FAVOURABLE PATIENTS ========== ############
    
    ### COLLECT PATIENTS, WHO ONLY HAVE FAVOURABLE OUTCOMES AT END OF TREATMENT & AT ALL FOLLOW-UP TIMEPOINTS 
    #. ==>TB-1021: 12 & 18 MONTHS, TB-1022: 18 & 24 MONTHS

    outcome_df=pd.read_csv('../data/tb_1018_20_21_22_30_outcome.csv.gz',index_col=0)
    outcome_df=outcome_df.set_index('USUBJID',drop=True)
    outcome_df=outcome_df.rename(columns={'UNFAVOURABLE_OUTCOME_CATEGORY_AT_18_MONTHS':'UNFAVOUR_CAT_AT_18_MONTHS'})
    
    df_=outcome_df.reset_index()#
    df_['STUDYID']=df_['USUBJID'].str.split('/',expand=True)[0].values
    df_=df_.set_index('USUBJID')
    df_=df_.loc[X_subset['USUBJID'].unique()]
    
    pats_with_fav=[]
    for study,d in df_[df_['STUDYID'].isin(['TB-1022','TB-1021'])].groupby('STUDYID'):
        if study=='TB-1021':
            out_cols=['RESULT_AT_END_OF_TREATMENT','RESULT_AT_12_MONTHS','RESULT_AT_18_MONTHS']#,'RESULT_AT_24_MONTHS']
            pats_with_fav.extend(d.loc[(d[out_cols]=='FAVOURABLE').all(axis=1),:].index.tolist())
            
        if study=='TB-1022':
            out_cols=['RESULT_AT_END_OF_TREATMENT','RESULT_AT_18_MONTHS','RESULT_AT_24_MONTHS','UNFAVOURABLE_OUTCOME_CATEGORY_AT_24_MONTHS']
            pats_with_fav.extend(d.loc[(d[out_cols]=='FAVOURABLE').all(axis=1),:].index.tolist())
    
    len(pats_with_fav)
    
    
    #### ================================ UNFAVOURABLE PATIENTS ========== ############
    
    ### COLLECT PATIENTS, WHO HAVE FAVOURABLE OUTCOME AT END OF TREATMENT, BUT HAVE AT LEAST ONE UNFAVOURABLE OUTCOME AT ANY OF THE FOLLOW-UP TIMEPOINTS 
    #. ==>TB-1021: 12 & 18 MONTHS, TB-1022: 18 & 24 MONTHS
    
    pats_with_unfav=[]
    
    for study,d in df_[df_['STUDYID'].isin(['TB-1022','TB-1021'])].groupby('STUDYID'):
        if study=='TB-1021':
            out_cols=['RESULT_AT_12_MONTHS','RESULT_AT_18_MONTHS']#,'RESULT_AT_24_MONTHS']
    
            pats_=d.loc[(d['RESULT_AT_END_OF_TREATMENT']=='FAVOURABLE')&(d[out_cols]=='UNFAVOURABLE').any(axis=1),:].index.tolist()
            pats_with_unfav.extend(pats_)
            
        if study=='TB-1022':
            out_cols=['RESULT_AT_18_MONTHS','RESULT_AT_24_MONTHS']#,'UNFAVOURABLE_OUTCOME_CATEGORY_AT_24_MONTHS']
            
            pats_=d.loc[(d['RESULT_AT_END_OF_TREATMENT']=='FAVOURABLE')&\
                  (d[out_cols]=='UNFAVOURABLE').any(axis=1)&\
                  (d['UNFAVOURABLE_OUTCOME_CATEGORY_AT_24_MONTHS']!='FAVOURABLE')&\
                  (~d['UNFAVOURABLE_OUTCOME_CATEGORY_AT_24_MONTHS'].isna())\
                    ,:].index.tolist()
            
            pats_with_unfav.extend(pats_)
    
    
    ## USING THE RAW DISPOSITION EVENTS DATAFRAME (ds) &  PREPROCESSED de DATAFRAME (day of disposition events extracted/patient),
    #. EXTRACT PATIENT IDS:
    #  ==> WHO HAVE DOCUMENTED RELAPSE OR TREATMENT FAILURE (
    #. ==> WHOSE RELAPSE OR TREATMENT FAILURE IS AFTER THE LAST DAY OF THE INITIAL THERAPY
    
    
    #####=========  1. Read the dataframes
    ds=pd.read_csv('../../C-Path_data/fullExportDb-1025-Member-CSV/ds.csv',low_memory=False)
    ds=ds.loc[ds['USUBJID'].isin(X_subset['USUBJID'].unique())]
    
    de=pd.read_csv('../../C-Path_data/preprocessing/disposition_events.csv',low_memory=True) 
    de=de.set_index('USUBJID')
    
    
    
    #####=========  2. Extract Patients, whose relapse is after the last day available in the initital dataset (==relapse is after initital therapy completion)
    
    # 3 patients didn't have an exact day of relapse==> either 'YES' or 'FOLLOW-UP PHASE' was extracted during preprocessing
    #. => drop these rows temporarily, as they are string values, this way the other RELAPSE days can be converted to float
    de_=de[(~de['RELAPSE'].isin(['YES','FOLLOW-UP PHASE']))&(~de['REINFECTION'].isin(['YES','FOLLOW-UP PHASE']))]
    de_['RELAPSE']=de_['RELAPSE'].astype(float)
    
    ds_rel=(ds.loc[ds['USUBJID'].isin(pats_with_unfav)].groupby('USUBJID').apply(lambda x: x.loc[(x['DSDECOD'].str.contains('RELAPSE'))|((x['DSTERM'].str.contains('RELAPSE')))]))
    #print(ds_rel['DSDECOD'].value_counts(dropna=False))
    
    
    # Extract the last day available in the concatenated clinical & drug regimen dataset ==> this day will be assumed to be the completion day of the initial therapy
    ## N.B.: For TB-1021
    max_days=X_subset.loc[X_subset['USUBJID'].isin(pats_with_unfav),:].groupby('USUBJID').apply(lambda x: x['DAY'].max())
    
    ## Subset de to the unfavourable patient ids
    com_idx=list(set(de_.index)&set(pats_with_unfav))
    de_=de_.loc[(com_idx)]
    
    comm_idx=list(set(de_.index)&set(max_days.index))
    
    ## Add the last day values to the de dataframe
    de_=pd.concat([max_days,de_.loc[comm_idx,:]],axis=1)
    
    ## Add ARMS to the de dataframe
    arms=X_subset.groupby('USUBJID').apply(lambda x:x['ARM'].unique()[0])
    de_['ARM']=arms.loc[de_.index]
    
    ## Column 0 contains tha last days coming from the clinical data
    # => For TB-1021: Last day of therapy (==COMPLETION CONTINUATION PHASE) is available for some patients
    # => Take the maximum of the last day of therapy coming frok clinical data & COMPLETION CONTINUATION PHASE in these cases, as clinical data is sparse
    de_['last_therapy_day']=de_[[0,'COMPLETION CONTINUATION PHASE']].max(axis=1)
    de_.loc[~de_['COMPLETION CONTINUATION PHASE'].isna(),'last_therapy_day']=de_.loc[~de_['COMPLETION CONTINUATION PHASE'].isna(),'COMPLETION CONTINUATION PHASE'].values
    
    
    ## Extract those patients, where relapse was observed after the completion of the initial therapy
    relapse_after_obs_period=de_[(de_['last_therapy_day']<de_['RELAPSE'])]
    
    ## For TB-1021 Study: for the 4 month arms, the continuation phae goes up until 6 months (patients taking placebo after month 4)
    #. => Extract those relapses, that occurred after completion of the 4 month therapy, but before the the end of the 6 month study observation period
    #. => 
    relapse_during_obs_period=de_[(de_['last_therapy_day']>=de_['RELAPSE'])\
                                &(de_['RELAPSE']>100)\
                                &(de_['ARM'].str.contains(r'2MHRZ/2MHR|2EMRZ/2MR'))]
    
    
    #####=========  3. Extracting the IDs of the 3 patients, where the relapse data as either missing, (2 TB-1022 patients), 
    #.                 or it can be imputed from the treatment restart day (TB-1021/2003995)
    idx=de[(de.index.isin(pats_with_unfav))\
        &(de['RELAPSE'].isin(['YES','FOLLOW-UP PHASE']))].index

    
    #print('set(idx)&set(de_.index)',set(idx)&set(de_.index))
    
    pats_with_sparse_relapse_data=pd.DataFrame({'RELAPSE':[238,np.nan,np.nan],
                                                  'ARM':arms.loc[idx],
                                                 'last_therapy_day':de_.loc[idx,'last_therapy_day'].values,
                                                  'STUDYID':de[(de.index.isin(pats_with_unfav))\
                                                                &(de['RELAPSE'].isin(['YES','FOLLOW-UP PHASE']))]['STUDYID'].tolist()},
                                                 index=idx)
    #pats_with_sparse_relapse_data['last_therapy_day']=de_.loc[set(idx)&set(de_.index)],
    
    #####========= 4. Concatenate all relapse patients into 1 dataframe
    #print('relapse_during_obs_period',relapse_during_obs_period)
    pats_with_relapse_df=pd.concat([relapse_after_obs_period[['RELAPSE','STUDYID','ARM','last_therapy_day']],\
                               relapse_during_obs_period[['RELAPSE','STUDYID','ARM','last_therapy_day']],\
                               pats_with_sparse_relapse_data[['RELAPSE','STUDYID','ARM','last_therapy_day']]],axis=0)
    
    pats_with_relapse_df.columns=['RELAPSE_DAY','STUDYID','ARM','last_therapy_day']
    pats_with_relapse_df['RELAPSE']=1


    ## Create dataframe for favourable patients
    max_days_fav=X_subset.loc[X_subset['USUBJID'].isin(pats_with_fav),:].groupby('USUBJID').apply(lambda x: x['DAY'].max())
    arms_fav=X_subset.loc[X_subset['USUBJID'].isin(pats_with_fav),:].groupby('USUBJID').apply(lambda x: x['ARM'].unique()[0])
    study_fav=X_subset.loc[X_subset['USUBJID'].isin(pats_with_fav),:].groupby('USUBJID').apply(lambda x: x['STUDYID'].unique()[0])
    
    pats_wo_relapse_df = pd.DataFrame({'RELAPSE':0},index=pats_with_fav)
    pats_wo_relapse_df['last_therapy_day'] = max_days_fav.loc[pats_wo_relapse_df.index].values
    pats_wo_relapse_df['ARM'] = arms_fav.loc[pats_wo_relapse_df.index].values
    pats_wo_relapse_df['STUDYID'] = study_fav.loc[pats_wo_relapse_df.index].values
    
    pats_relapse_df = pd.concat([pats_with_relapse_df,pats_wo_relapse_df],axis=0)

    pats_relapse_df.index.name='USUBJID'



    
    ## FOR SOME PATIENTS, RETREATMENT DURING FOLLOW-UP STARTED EARLIER AS THE RELAPSE_DAY IN THE DISPOSITION EVENTS
    ## ==> TAKE THE FIRST DAY OF RETREATMENT AS RELAPSE DAYS FOR THESE PATIENTS
    ex = pd.read_csv('../../C-Path_data/fullExportDb-1025-Member-CSV/ex.csv', low_memory=False)
    ex = ex[ex['USUBJID'].isin(pats_relapse_df.index.tolist())]
    
    ## Extract patients with retreatmetn during follow-up
    retreatment=ex.loc[ex['EPOCH'].str.contains('FOLLOW',na=False),:].groupby('USUBJID').apply(lambda x: x['EXSTDY'].min()).sort_index().to_frame()
    retreatment_idx=ex.loc[ex['EPOCH'].str.contains('FOLLOW',na=False),'USUBJID'].unique()
    a=pats_relapse_df.loc[retreatment_idx,'RELAPSE_DAY'].sort_index()
    
    ## Concatenate retreatment start day with relapse day coming from disposition events
    b = pd.concat([retreatment,a],axis=1)
    b.columns=['retreatment_start','RELAPSE_DAY']
    
    ## For patients where retreatment started earlier then relapse_day, take the retreatment day as their relapse day
    retreatment_earlier_than_relapse = b[(b['RELAPSE_DAY'] - b['retreatment_start'])>0].index.tolist()
    pats_relapse_df.loc[retreatment_earlier_than_relapse,'RELAPSE_DAY'] = b.loc[retreatment_earlier_than_relapse,'retreatment_start'].values




    ### EXTRACT THE LAST DAY OF THERAPY DRUG ADMINISTRATION USING THE DR_REG DATAFRAME
    #  => FOR SOME REMOXTB PATIENTS, LAST DAY OF THERAPY WAS TAKEN DOWN AS LAST DAY PALCEBO WAS APPLIED
    #  => INSTEAD, EXXTRACT LAST DAY WHERE NON-PLACEBO THERAPY DRUGS WERE APPLIED, TO GET A BETTER SENSE OF RELAPSE AFTER EOT
    month_4_idx = pats_relapse_df[~pats_relapse_df['ARM'].str.contains('Control|2EHRZ')].index.tolist()

    t=pd.read_csv('../data/out_temporal_pat_regimens_1018_20_21_22_30.csv.gz',low_memory=False,index_col=0)
    t = t[t['USUBJID'].isin(month_4_idx)]
    
    max_days = {}


    ## Loop over 4-month patients, and extract the last day of therapy drug adpplication before relapse or if there was no relapse,
    #  take the last day of therapy drug application (250 days are set as a threshold to include patients with extendedn baseline therapy)
    
    for pat_id in (month_4_idx[:]):        

        rel_day = pats_relapse_df.loc[pat_id,'RELAPSE_DAY']#.values

    
        #print('rel_day',rel_day)
    
        ## If no relapse, take the 250 as threshold
        if str(rel_day)=='nan':
            try:
                thr_day = 250 #float(compl_day)
            except ValueError:
                #print(f'VAlueERror: {rel_day} is Nan!')
                continue
    
        ## If relapse, take the relapse day as threshold
        if str(rel_day)!='nan':
            
            try:
                thr_day = np.min([float(rel_day),250])
                
            except ValueError:
                #print(f'VAlueERror: {rel_day} is not Nan!')
                continue
    
        #print('thr_day',thr_day)
        t_ = t[(t['USUBJID']==pat_id) & (t['DAY']<thr_day)]

        ## If there is drug regimen information, extract the maximal number of non-pacebo drugs taken, and then extract the first day where the meximum dose
        #  was reached ==> this was the last day therapy drugs were applied
        if t_.shape[0]>0:
            #print(t_.loc[:,t_.columns.str.contains('num_of_doses')].max().max())
            num_of_doses= t_.loc[:,(t_.columns.str.contains('num_of_doses'))&\
                                   (~t_.columns.str.contains('placebo'))].max().sort_values()
            coln,val = num_of_doses.tail(1).index, num_of_doses.tail(1).values[0]
            last_ther_day = t_.loc[(t_[coln]==val).values,'DAY'].iloc[0]

            max_days[pat_id]=last_ther_day
    
    
    max_days=pd.DataFrame(index=max_days.keys(),data=max_days.values())

    ## For patients
    pats_relapse_df.loc[max_days.index,'last_therapy_day'] = max_days[0].values

    del t



    pats_relapse_df['DAYS_BETWEEEN_THERAPY_END_AND_RELAPSE']=(pats_relapse_df['RELAPSE_DAY'] - pats_relapse_df['last_therapy_day']).values

    if include_rifaquin==True:
        ## ADD RIFAQUIN RELAPSE PATIENTS
        rif_rel = extract_rifaquin_relapse()
        pats_relapse_df = pd.concat([pats_relapse_df,rif_rel[pats_relapse_df.columns]],axis=0)



    def replace_arm_names(d):
    
        d['ARM'] = d['ARM'].replace({'Gati-arm regimen (4 month regimen)':'Gatifloxacin',
                                 'Control-arm regimen (6 month regimen)':'Control',
                                  'Control Regimen : 2 months of daily ethambutol, isoniazid, rifampicin, and pyrazinamide followed by 4 months of daily isoniazid and rifampicin.':'2EHRZ/4HR_RIF',
                                'Study Regimen 2: 2 months of daily ethambutol, moxifloxacin, rifampicin, and pyrazinamide followed by 4 months of once weekly moxifloxacin and rifapentine.':'2EMRZ/4MP',
                                 'Study Regimen 1: 2 months of daily ethambutol, moxifloxacin, rifampicin, and pyrazinamide followed by 2 months of twice weekly moxifloxacin and rifapentine.':'2EMRZ/2MP'},regex=False)
        return d
    
    pats_relapse_df = replace_arm_names(pats_relapse_df)
    
    return pats_relapse_df#,relapse_during_obs_period,relapse_after_obs_period,pats_with_sparse_relapse_data,max_days



def extract_last_init_therapy_day_from_drug_regimen(pat_ids):

    ## Load drug regimen data, containing the daily taken doses
    dr_reg=pd.read_csv('../data/out_temporal_pat_regimens_1018_20_21_22_30.csv.gz',index_col=0)
    dr_reg=dr_reg[dr_reg['USUBJID'].isin(pat_ids)]
    
    ## Drop placebo columns ==> extract last day when TB therapy was applied 
    ## REMOXTB: End-of-therapy outcome was determined like this (REMOXTB manual: 11.3 Other secondary ednpoints, page 157/253)
    ## OFLOTUB: no placebos applied 
    dr_reg=dr_reg.loc[:,~dr_reg.columns.str.contains('placebo')].copy()
    
    ## Extract colnames containing the drug names
    drug_cols=[c.split('_cumulative_dose')[0] for c in dr_reg.columns if '_cumulative_dose' in c]
    
    ## IF drug was not taken anymore, the daily dose is set to 0. Drop all drug columns which have only zeroes (meaning patient didn't take them),
    #. replace 0s with NaNs, and drop all rows, which only have NaNs in the drug column 
    #. ==> The last day where drug was applied is the last day of initial therapy
    last_init_therapy_days=dr_reg.loc[:,['DAY','USUBJID']+drug_cols].groupby('USUBJID').apply(lambda x:x.replace(0,np.nan).dropna(subset=drug_cols,how='all',axis=0)['DAY'].max())
    
    ## Drop days without drug application from dr_reg for later use
    dr_reg_appl_days=dr_reg.loc[:,['DAY','USUBJID']+drug_cols].groupby('USUBJID',as_index=False).apply(lambda x:x.replace(0,np.nan).dropna(subset=drug_cols,how='all',axis=0))
    
    ### PATIENTS WITH TREATMENT GAPS: DRUG REGIMEN DATA ALSO CONTAINS RETREATMENTS FOLLOWING THE INITIAL TREATMENT IN THE DATAFRAME ==> EXTRACT LAST DAY OF INIT. THERAPY
    ## Extract patients, who had treatment gaps 
    #  ==> not all days are consecutive in the patient's drug regimen dataframe, 
    #  ==> if difference of 'DAY' value between two consecutive rows is larger than 1==> therapy gap & retreatment
    # IMPORTANT: For some patients, treatment went on well into the follow-up period (>250 days), without a treatment gap! 
    #.           For these patients, consider the very last day as last day of treatment, not the last day of the originally scheduled continuation period
    day_diffs=dr_reg.loc[:,['DAY','USUBJID']].groupby('USUBJID').apply(lambda x:x['DAY'].diff().max()).sort_values(ascending=False)
    pats_with_treatment_gaps=day_diffs[day_diffs>1].index.tolist()
    
    
    if len(pats_with_treatment_gaps)>0:
    
        ## For patients with treatment gaps, extract the first day after the treatment gap
        #print(dr_reg.loc[dr_reg['USUBJID'].isin(pats_with_treatment_gaps),['DAY','USUBJID']].groupby('USUBJID',as_index=True).apply(lambda x:\
        #                                                                                                               x.loc[x['DAY'].diff()>1,'DAY']))
        retreat_start_day=dr_reg.loc[dr_reg['USUBJID'].isin(pats_with_treatment_gaps),['DAY','USUBJID']].groupby('USUBJID',as_index=True).apply(lambda x:\
                                                                                                                       x.loc[x['DAY'].diff()>1,'DAY']).sort_values(ascending=False)
        retreat_start_day = retreat_start_day.reset_index().drop(columns=['level_1']).set_index('USUBJID')
    
        ## Using the retreat_start_day, extract the last day of the initial therapy for patients with therapy gaps
        last_init_day_of_pats_with_retreatment=dr_reg_appl_days.loc[dr_reg_appl_days['USUBJID'].isin(pats_with_treatment_gaps), ['DAY', 'USUBJID']].groupby('USUBJID', as_index=True).apply(
                                                            lambda x: x.loc[x['DAY'] < retreat_start_day.loc[x['USUBJID'].unique()[0]].values[0], 'DAY'].max()
                                                            ).sort_values(ascending=False)
    
        
        ## Replace the last day values of patients with treatment gaps with the extracted last day values
        last_init_therapy_days.loc[last_init_day_of_pats_with_retreatment.index]=last_init_day_of_pats_with_retreatment.values

    return last_init_therapy_days

##========================================= 
def subset_pats_with_therapy_in_period(period_num,period_end_days,last_init_ther_days,
                                       pats_with_relapse_df,period_end_day,X,outcome_label,data_param_key):
    
    ## SUBSET TO PATIENTS WHO WERE TAKING DRUGS DURING THE PERIOD
    #if parameters_for_analysis[data_param_key]['result_cat']!='RELAPSE': 
        
    ## Drop patients whose last therapy day was before the previous period_end_day 
    #  i.e. period_end_day=125, keep patients having last therapy day later than 93 +5 days (some leeway)
    if period_num==1:
        pat_ids_=last_init_ther_days[~(last_init_ther_days<=period_end_days[period_num])].index.tolist()
    
    if period_num>1:
        pat_ids_=last_init_ther_days[~(last_init_ther_days<=period_end_days[period_num-1]+5)].index.tolist()

    ## If perios is baseline, keep all the patients
    if period_num==0:
        pat_ids_=last_init_ther_days.dropna().index.tolist()
    
    
    ## SUBSET TO PATIENTS WHO WERE HAD THEIR RELAPSE AFTER THE END OF PERIOD & PATIENTS WITHOUT RELAPSE 
    ## (RELAPSE DAY IS NAN + 2 PATIENTS WITH RELAPSE IN FOLLOW-UP PERIOD, BUT UNKNOWN EXACT RELAPSE DAY)
    if parameters_for_analysis[data_param_key]['result_cat']=='RELAPSE' or 'pred_prob' in outcome_label:
        
        if isinstance(period_end_day,int):#!='all':
            pat_ids__ = pats_with_relapse_df[(pats_with_relapse_df['RELAPSE_DAY']>period_end_day)\
                                            |(pats_with_relapse_df['RELAPSE_DAY'].isna())]['RELAPSE_DAY'].index.tolist() #
    
        if period_end_day=='all':
            print(period_end_days[period_num-1])
            pat_ids__ = pats_with_relapse_df[(pats_with_relapse_df['RELAPSE_DAY']>182)\
                                            |(pats_with_relapse_df['RELAPSE_DAY'].isna())]['RELAPSE_DAY'].index.tolist() #

        if period_end_day=='baseline':
            pat_ids__ = pats_with_relapse_df[(pats_with_relapse_df['RELAPSE_DAY']>0)\
                                             |(pats_with_relapse_df['RELAPSE_DAY'].isna())]['RELAPSE_DAY'].index.tolist()
        
        ## CONSIDER ONLY PATIENTS WHO RECEIVED THERAPY SINCE THE LAST PERIOD
        pat_ids_=list(set(pat_ids_)&set(pat_ids__))


    print('pat_ids_ before arm check',len(pat_ids_))
    
    ### FINAL CHECK 
    ## FOR EOT OUTCOME PREDICTION: 
    #.  => ONLY CONSIDER BASELINE - PENULTIMATE MONTHE, AS OUTCOME LABELS WERE DETERMINED AT LAST MONTH (4 OR 6), 
    #.     THEREFORE PREDICTION DOESN'T MAKE SENSE THERE
    ## FOR RELAPSE PREDICTION: 
    #. => IF PERIOD IS 4 MONTHS OR LESS, KEEP ALL PATIENTS
    #. => IF PERIOD IS OVER 4 MONTHS KEEP ONLY PATIENTS IN ARMS WITH 6 MONTHS OF TB DRUG APPLICATION    

    month_4_arms=['Gatifloxacin','2MHRZ/2MHR', '2EMRZ/2MR']
    month_6_arms=['Control','2EHRZ/4HR']

    if outcome_label=='RESULT_AT_END_OF_TREATMENT':
        month4_periods=['baseline',31,62,93,125][:-1]
        month6_periods=[125,160,'all'][:-1]

    if outcome_label=='RELAPSE' or 'pred_prob' in outcome_label:
        month4_periods=['baseline',31,62,93,125]
        month6_periods=[160,'all']
        
    
    if period_end_day in month4_periods:
        arms_to_consider= month_6_arms + month_4_arms
        arm_pats=X.loc[X['ARM'].isin(arms_to_consider),'USUBJID'].unique().tolist()
        pat_ids_=list(set(pat_ids_)&set(arm_pats))
        
    if period_end_day in month6_periods:
        arms_to_consider= month_6_arms
        arm_pats=X.loc[X['ARM'].isin(arms_to_consider),'USUBJID'].unique().tolist()
        #print(pat_ids_)
        pat_ids_=list(set(pat_ids_)&set(arm_pats))

    ## Return empty list no patients should be considered (EOT outcome prediction, 6 months)
    if period_end_day not in month6_periods and period_end_day not in month4_periods:
        pat_ids_=[]
    
    return pat_ids_

####======================================================================
## Calculate cumulatie toxicity grade of adverse events \
## + ## Calculate the rolling average number of visits, the clinical event was observed
def calucate_cumul_adverse_clinical_events(X_subset,period_end_day):
    
    cols_to_keep=select_temporal_cols_with_suff_pat_data('ae',X_subset,0.04)
    cols_to_keep
    
    # List of strings you want to match (e.g., from 'USUBJID' list)
    USUBJID = X_subset['USUBJID'].unique()
    
    # Read only specific columns
    cols_to_keep_=['USUBJID','DAY','STUDYID'] + cols_to_keep
    
    # Use `chunksize` to read the file in chunks and filter rows for patients in dataset
    filtered_rows = []
    
    fn=os.path.join('../data','out_ae_standardised_temporal.csv.gz')
    
    for chunk in pd.read_csv(fn, usecols=cols_to_keep_, chunksize=10000):
        # Filter rows where 'USUBJID' column contains values from the `USUBJID` list
        filtered_chunk = chunk[chunk['USUBJID'].isin(USUBJID)]
        filtered_rows.append(filtered_chunk)
    
    # Concatenate the filtered chunks into a single DataFrame
    ae_df = pd.concat(filtered_rows, ignore_index=True)

    ## Sort by patient ID and study day
    ae_df=ae_df.sort_values(['USUBJID','DAY'])

    ## Subset to period considered, if period is numeric. 
    ## If not numeric (baseline or all day), consider all data points
    if isinstance(period_end_day, str)==False:
        ae_df=ae_df[ae_df['DAY']<=period_end_day]



    ## Calculate the cumulative toxicity grade over the days for each patient
    ae_cum_df = ae_df.sort_values(['USUBJID','DAY']).groupby(['USUBJID','STUDYID']).apply(lambda x: x.loc[:,x.columns.str.startswith('ae_')].fillna(0).cumsum())
    ae_cum_df['DAY']=ae_df['DAY'].values

    ## Set patient ID and Day as index + rename the columns to reflect the variable's nature
    ae_cum_df=ae_cum_df.reset_index().drop(columns=['STUDYID','level_2']).set_index(['USUBJID','DAY'])
    ae_cum_df.columns=[f'{col}_cumul_toxgrade' for col in ae_cum_df.columns]

    ## Sort main dataframe by patient and study day and set them as index
    ## ==> Prepare to merge the cumulative AE dataframe on the patient IDs and days of the main dataframe
    X_subset__ = X_subset.sort_values(['USUBJID','DAY']).set_index(['USUBJID','DAY'])

    ## MERGE main dataframe with the cumulative AE tox.grade dataframe
    ## ==> how='left' means, only those days and patients rows are merged to the X_subset__ dataframe, which were exisitng in the ae_cum_df as well
    c=pd.merge(X_subset__, ae_cum_df, how="left", on=['USUBJID','DAY'])
    
    ## Forward fill the cumulative AE variables to fill up the enevtaul gaps
    ## + fill the remaining NaNs with 0==> for these visits or prior visits to them, no AE events were recorded
    c.loc[:,c.columns.str.endswith('_cumul_toxgrade')]=c.groupby('USUBJID',as_index=False).apply(lambda x:x.loc[:,c.columns.str.endswith('_cumul_toxgrade')].ffill()).fillna(0).values


    #### CE COLUMNS
    ## DEPRECATED  -Calculate cumulative sum of the CE variables
    ## ==> The cumulative vairable will reflect on how many visits the clinical event was recorded for the patient 
    #c.loc[:,c.columns.str.startswith('ce_')] = c.groupby('USUBJID',as_index=False).apply(lambda x:x.loc[:,c.columns.str.startswith('ce_')].cumsum()).values
    #c.columns=[coln.replace('STD_CAT_ORDINAL_RESULT','cumul_visit') if coln.startswith('ce_') else coln for coln in c.columns]

    ## Calculate the rolling average number of visits, the clinical event was observed
    ## ==> This will normalise between patients who have less visits
    c.loc[:,c.columns.str.startswith('ce_')] = c.groupby('USUBJID',as_index=False).apply(lambda x:x.loc[:,c.columns.str.startswith('ce_')].cumsum().div(np.arange(1,len(x)+1),axis=0)).values
    c.columns=[coln.replace('STD_CAT_ORDINAL_RESULT','avg_visits_obs') if coln.startswith('ce_') else coln for coln in c.columns]
    c.loc[:,c.columns.str.endswith('avg_visits_obs')]=c.loc[:,c.columns.str.endswith('avg_visits_obs')].replace([np.inf, -np.inf], np.nan)
    
    ae_cumul_colnames=c.columns[c.columns.str.endswith('cumul_toxgrade')].tolist()

    return c.reset_index(),ae_cumul_colnames


####======================================================================
## CONVERT THE CONTINUOUS RELAPSE DAYS VARIABLES TO CATEGORICAL BASED ON PREDEFINED TIME INTERVALS
##. i.e. early relapse:1 (<365 days after therapy end) vs. no relapse+ late relapse
def cut_relapse_days_to_interval_categories(pats_relapse_df,data_param_key):

    bins=parameters_for_analysis[data_param_key]['bins']
    labels=parameters_for_analysis[data_param_key]['labels']

    ## Drop 2 patients, whose relapse day information is missing
    pats_to_drop=pats_relapse_df.loc[(~pats_relapse_df['RELAPSE']==1)&(pats_relapse_df['RELAPSE_DAY'].isna()),:].index.tolist()

    ## Drop a handful patients who had a relapse label during therapy phase
    pats_to_drop_=pats_relapse_df.loc[pats_relapse_df['DAYS_BETWEEEN_THERAPY_END_AND_RELAPSE']<0,:].index.tolist()
    #print(len(pats_to_drop),len(pats_to_drop_))

    pats_to_drop=pats_to_drop_ + pats_to_drop
    
    pats_relapse_df=pats_relapse_df.drop(index=pats_to_drop)

    ## Cut the intervals
    pats_relapse_df['RELAPSE'] = pd.cut(pats_relapse_df['DAYS_BETWEEEN_THERAPY_END_AND_RELAPSE'],bins=bins,labels=labels).astype(str)

    #print(pats_relapse_df['RELAPSE'].value_counts())

    ## For non relapssing patients, set 0 as label
    pats_relapse_df.loc[pats_relapse_df['RELAPSE']=='nan','RELAPSE']=0
    pats_relapse_df['RELAPSE']=pats_relapse_df['RELAPSE'].astype(int)

    return pats_relapse_df



####======================================================================
def replace_drug_cumul_columns_with_weight_norm_cumul_columns(X):
    
    ## Load dataframe with weight normalised cumulative drug doses applied
    dr_reg=pd.read_csv('../data/out_temporal_pat_regimens_weight_norm_1018_20_21_22_30.csv.gz',low_memory=False,index_col=0)

    ## Extract the weight normalised cumul. colnames
    weight_norm_cols=['USUBJID','DAY'] + dr_reg.columns[dr_reg.columns.str.endswith('_cumulative_dose_weight_norm')].tolist()

    ## Extract the absolute cumul. colnames from X, that should be replaced
    abs_cumul_cols=[col for col in  X.columns[X.columns.str.endswith('_cumulative_dose')] if 'placebo' not in col]

    ## Merge on DAY and USUBJID
    X_ = pd.merge(X,dr_reg[weight_norm_cols],how='left',on=['USUBJID','DAY'])

    ## Forward and backfill the weight norm columns for days, where the drug was not applied
    X_[weight_norm_cols] = X_.groupby('USUBJID').apply(lambda x: x.sort_values('DAY')[weight_norm_cols].ffill().bfill()).values

    ## Drop the columns with absolute cumul doses
    X_=X_.drop(columns=abs_cumul_cols)

    return X_


####======================================================================
def replace_cumul_day_of_appl_with_adherence(X):
    
    scheduled_doses={'2EHRZ/4HR':182, # 8+9+9 weeks , 7 days /week
                    '2MHRZ/2MHR':119,# 8+9 weeks , 7 days /week
                    '2EMRZ/2MR':119,# 8+9 weeks , 7 days /week
                    'Gatifloxacin':102, # 4 months (==8+9 weeks) , 6 days /week
                    'Control':156  # 6 months (26== weeks) , 6 days /week
                    } 
    
    
    l=[]
    for arm,arm_df in X.groupby('ARM'):
        arm_df['dr_reg_study_drugs_cumul'] = arm_df['dr_reg_study_drugs_cumul']/scheduled_doses[arm]
        l.append(arm_df)
    
    
    X_=pd.concat(l,axis=0)
    X_=X_.sort_values(['USUBJID','DAY'])

    return X_


###======================================================================
## Load preprocessed-imputed data, and modify the variables (add or drop) depending on the prediction setup, which is contained at the 
#. end of the "data_param_key" variable
def load_and_modify_preprocessed_data(data_param_key):
    
    ## Define X and y dataframes for training
    fn='../data/'+parameters_for_analysis[data_param_key]['fn']+'_preproc_data_with_imp.csv.gz'
    X=pd.read_csv(fn,index_col=0)
    X=X.rename(columns=lambda x: x.replace('<', 'lower than'))
    X=X.rename(columns=lambda x: x.replace('>', 'higher than'))

    if 'MGIT_LJ_disc' in data_param_key:
        X['mb_MGIT_LJ_discordant']=0
        X.loc[(X['mb_MGIT_STD_RESULT']==1)&(X['mb_LJ-culture_STD_RESULT']==0),'mb_MGIT_LJ_discordant']=1
        X__=X.sort_values(by=['USUBJID','DAY']).copy()
        X__['mb_MGIT_LJ_discordant'] = X__.groupby('USUBJID').apply(lambda x:x.sort_values('DAY')['mb_MGIT_LJ_discordant'].cumsum()).values
        X=X__.loc[X.index,:].copy()

    
    if 'RACE' not in X.columns:
        X['RACE']=X['USUBJID'].map(race_dict)
        X = pd.concat([X.drop(columns=['RACE']),pd.get_dummies(X['RACE'],dtype=int,prefix='RACE')],axis=1)
        race_colnames=X.columns[X.columns.str.contains('RACE_')].tolist()

    ## One-hot encode 'SEX'
    X['SEX'] = X['SEX'].replace({0:'F',1:'M'})
    X = pd.concat([X.drop(columns=['SEX']),pd.get_dummies(X['SEX'],dtype=int,prefix='SEX')],axis=1)
    X=X.drop(columns=['SEX_F'])

    ## One-hot encode 'ZN-smear oridinal column'
    #if 'mb_ZN-smear_STD_CAT_ORDINAL_RESULT' in X.columns:
    #    X = pd.concat([X.drop(columns=['mb_ZN-smear_STD_CAT_ORDINAL_RESULT']),\
    #                   pd.get_dummies(X['mb_ZN-smear_STD_CAT_ORDINAL_RESULT'].astype('Int64'),dtype=int,prefix='mb_ZN-smear')],axis=1)

    X=X.drop(columns=['mb_ZN-smear_STD_RESULT'])

    
    X['vs_BMI_STD_NUM_RESULT'] = (X['vs_Weight_STD_NUM_RESULT']/(X['vs_Height_STD_NUM_RESULT']/100)**2).values
    X['ARM']=X['ARM'].replace({'Gati-arm regimen (4 month regimen)':'Gatifloxacin',
                             'Control-arm regimen (6 month regimen)':'Control'},regex=False)

    X['therapy_arm_duration'] = '4-month'
    X.loc[X['ARM'].isin(['2EHRZ/4HR','Control']),'therapy_arm_duration'] = '6-month'

    ## One-hot encode 'ARM'
    if 'with_arm' in data_param_key:
        X = pd.concat([X,pd.get_dummies(X['ARM'],dtype=int,prefix='ARM')],axis=1)


    ## Replace extreme Hyperkalemia values with mean (probably false measurements)
    mean_K=X.loc[X['lb_Blood Potassium_STD_NUM_RESULT']<=12,'lb_Blood Potassium_STD_NUM_RESULT'].mean()
    X.loc[X['lb_Blood Potassium_STD_NUM_RESULT']>12,'lb_Blood Potassium_STD_NUM_RESULT']=mean_K

    ## Replace absolute cumulative doses with weight-normalised cumulative doses
    if 'weight_norm' in data_param_key:
        X = replace_drug_cumul_columns_with_weight_norm_cumul_columns(X)

    ## Keep only microbiological variables
    if 'mb_only' in data_param_key:
        X = X.loc[:,X.columns.str.contains(r'mb_|USUBJID|STUDYID|index|\bARM\b|\bDAY\b',regex=True)]

    ## Drop microbiological variables
    if 'without_mb' in data_param_key:
        X = X.loc[:,~X.columns.str.startswith('mb_')]
    
    ## Drop drug regimen variables, except for drug adherence (dr_cumul_dose)
    if 'without_dr_reg' in data_param_key:
        X = X.loc[:,~X.columns.str.endswith('cumulative_dose')]

    ## Divide cumulative days of application with the number of days scheduled ==> approximate drug adherence with a 0-1 number
    if 'with_adherence' in data_param_key:
        X = replace_cumul_day_of_appl_with_adherence(X)
    

    ## SPLIT THE CUMULATIVE DOSES OF THE PATIENTS BETWEEN THE ARMS ==> CONTROLLING FOR THE DIFFERENT NUMBER OF SCHEDULED DOSES BETWEEN ARMS
    if 'dr_reg_per_arm' in data_param_key:
        
        X_=X.copy()
        dr_cumul_colnames = X_.columns[X_.columns.str.startswith('dr_reg')].tolist()
        arms_names= X_['ARM'].unique()
        dr_cumul_colns_within_arm=[f'{dr}_{arm}' for dr in dr_cumul_colnames for arm in arms_names]
        X_[dr_cumul_colns_within_arm]=0
        
        
        for dr in dr_cumul_colnames:
            for arm in X_['ARM'].unique():
                X_.loc[X_['ARM']==arm,f'{dr}_{arm}']=X_.loc[X_["ARM"]==arm,dr].values
        
        ## Identify columns with all 0 values ==> 
        #. these are drugs that were not taken in the arm 
        #. => drop them along with the original dr_reg colnames
        regs_with_no_appl=(X_[dr_cumul_colns_within_arm]==0).all()[(X_[dr_cumul_colns_within_arm]==0).all()].index.tolist()
        cols_to_drop = dr_cumul_colnames + regs_with_no_appl
        
        X=X_.drop(columns=cols_to_drop)

    ## SPLIT THE CUMULATIVE DOSES OF THE PATIENTS BETWEEN THE ARMS ==> CONTROLLING FOR THE DIFFERENT NUMBER OF SCHEDULED DOSES BETWEEN ARMS
    if 'basic_vars' in data_param_key:

        basic_vars = ['AGE', 
                #'ce_SWEAT_STD_CAT_ORDINAL_RESULT',
                #'ce_COUGH_STD_CAT_ORDINAL_RESULT',
                #'ce_CHEST PAIN_STD_CAT_ORDINAL_RESULT',
                #'ce_FEVER_STD_CAT_ORDINAL_RESULT',
                #'ce_HAEMOPTYSIS_STD_CAT_ORDINAL_RESULT', 
                #'vs_Weight_STD_NUM_RESULT',
                #'vs_Temperature_STD_NUM_RESULT',
                #'vs_Diastolic Blood Pressure_STD_NUM_RESULT',
                #'vs_Systolic Blood Pressure_STD_NUM_RESULT',
                #'vs_Heart Rate_STD_NUM_RESULT', 
                #'vs_Height_STD_NUM_RESULT',
                #'lb_Blood Creatinine_STD_NUM_RESULT',
                #'lb_Blood Hemoglobin_STD_NUM_RESULT',
                #'lb_Blood Alanine Aminotransferase_STD_NUM_RESULT',
                #'lb_Blood Aspartate Aminotransferase_STD_NUM_RESULT',
                #'lb_Blood Potassium_STD_NUM_RESULT',
                #'lb_Blood Platelets_STD_NUM_RESULT',
                'mb_ZN-smear_STD_CAT_ORDINAL_RESULT', 
                'mb_LJ-culture_STD_RESULT',
                #'mh_DYSPNEA', 
                #'mh_FEVER', 
                #'mh_WEIGHT LOSS', 
                #'mh_COUGH', 
                #'mh_CHEST PAIN',
                #'mh_HAEMOPTYSIS', 
                #'mh_SWEAT', 
                'RACE_ASIAN', 
                'RACE_BLACK',
                'RACE_MIXED RACE OR COLOURED', 
                'RACE_OTHER', 
                'SEX_M',
                'vs_BMI_STD_NUM_RESULT',
          
                ]
        necerssary_vars=['DAY','USUBJID' ,'ARM','STUDYID','index']
        vars_to_keep = necerssary_vars + basic_vars
        
        X = X[vars_to_keep].copy()

    return X,race_colnames

#####=======================================
def select_visits_with_dual_thresholds(
    df, time_col, patient_col, cutoff_day, before_threshold, after_threshold,verbose=True):
    """
    Filters visits for each patient based on a time cutoff with thresholds before and after.
    This way we can adjust for patients, who visited the clinic a couple of days later than scheduled

    Logic:
    - If the visit is before the cutoff and lies within the before-threshold, keep all visits up to that visit:
    - If there are no post-cutoff visits, take all visits up to the last prior visit, doesn't matter if it lies within the before-threshold or not. 
    - If there are post-cutoff visits, but the last visit before cutoff lies outside of the before-threshold, take all visits up to the post-cutoff visit, 
        if the post-cutoff visit lies within the after threshold.
    
    Parameters:
        df (pd.DataFrame): Input data with multiple visits per patient.
        time_col (str): Column name containing visit times (numeric).
        patient_col (str): Column name identifying patients.
        cutoff_day (int or float): Time cutoff.
        before_threshold (int or float): Max days before cutoff to accept a visit.
        after_threshold (int or float): Max days after cutoff to accept a visit.
    
    Returns:
        filtered_df (pd.DataFrame): Filtered visits for each patient.
    """
    selected_visits = []
    excluded_patients = []

    for patient_id, group in df.groupby(patient_col):
        group_sorted = group.sort_values(time_col)
        group_sorted = group_sorted.copy()
        group_sorted["diff"] = group_sorted[time_col] - cutoff_day

        before = group_sorted[(group_sorted["diff"] <= 0)]
        after = group_sorted[(group_sorted["diff"] > 0)]

        # Case 1: visit before cutoff within threshold
        valid_before = before[before["diff"] >= -before_threshold]
        if not valid_before.empty:
            last_before_day = valid_before[time_col].max()
            keep = group_sorted[group_sorted[time_col] <= last_before_day]
            selected_visits.append(keep)
            continue

        # Case 2: no valid_before, but valid after
        valid_after = after[after["diff"] <= after_threshold]
        if not valid_after.empty:
            first_after_day = valid_after[time_col].min()
            keep = group_sorted[group_sorted[time_col] <= first_after_day]
            selected_visits.append(keep)
            continue

        # Case 3: no valid_before and no valid_after, but some visits before
        if not before.empty:
            last_before_day = before[time_col].max()
            keep = group_sorted[group_sorted[time_col] <= last_before_day]
            selected_visits.append(keep)
        else:
            excluded_patients.append(patient_id)

    filtered_df = pd.concat(selected_visits, axis=0).drop(columns='diff')

    if verbose:
        total_patients = df[patient_col].nunique()
        num_excluded_patients=len(excluded_patients)
        print(f"Excluded {num_excluded_patients} out of {total_patients} patients "
              f"({num_excluded_patients / total_patients:.1%})\n")
    
    return filtered_df




#####================
def return_predict_label_dataframe(parameters_for_analysis,data_param_key,X,
                                  outcome_df,outcome_label,model_names,time_origin):
    import copy
    
    ## If not RELAPSE shpuld be predicted, subset the patients according to the availbility of the outcome results
    if parameters_for_analysis[data_param_key]['result_cat']=='RESULT_AT_END_OF_TREATMENT':
        
        ## Extract patients who have their last therapy day before therapy_day_thr ==> these patient probably dropped out
        last_day_per_pat_df=X.sort_values(by=['DAY']).groupby('USUBJID').apply(lambda x: x.loc[x.index[-1],:])
        pat_ids=last_day_per_pat_df[last_day_per_pat_df['DAY']>therapy_day_thr]['USUBJID'].tolist()
        #pat_ids=X['USUBJID'].unique().tolist()
        
        ## Subset outcome dataframe to patient considered
        target_df=outcome_df.loc[pat_ids,outcome_label]
        y=target_df.loc[pat_ids].replace(label2id)
        outcome_label_=copy.deepcopy(outcome_label)
        
    if parameters_for_analysis[data_param_key]['result_cat']=='RELAPSE':
        #outcome_label='RELAPSE'

        if  'survival' not in parameters_for_analysis[data_param_key].keys():
            pats_with_relapse_df=extract_21_22_relapse_pats()
    
            pats_with_relapse_df = pats_with_relapse_df.loc[list(set(X['USUBJID'].unique())&set(pats_with_relapse_df.index))]
    
            ## Create new prediction labels (or even multilabels) in the "RELAPSE" column based on the relapse day intervals defined in "bins" 
            if 'bins' in parameters_for_analysis[data_param_key].keys():
                pats_with_relapse_df = cut_relapse_days_to_interval_categories(pats_with_relapse_df,data_param_key)
            
            pat_ids=pats_with_relapse_df.index.tolist()
            target_df=pats_with_relapse_df[[outcome_label]]
            y=pats_with_relapse_df[[outcome_label]] 
            outcome_label_=copy.deepcopy(outcome_label)

        if 'survival' in parameters_for_analysis[data_param_key].keys() and parameters_for_analysis[data_param_key]['survival']==True:
            
            pats_with_relapse_df=extract_21_22_relapse_pats()
            
            ### LOAD AND EXTRACT REAL COMPLETION DAYS OF PATIENTS RECORDED
            de=pd.read_csv('../../C-Path_data/preprocessing/disposition_events.csv',low_memory=True) 
            de=de.set_index('USUBJID')
            comm_idx = list(set(de.index) & set(pats_with_relapse_df[pats_with_relapse_df['RELAPSE_DAY'].isna()].index))
            de_ = de.loc[comm_idx,:]
            
            compl_day_df= de_[~(de_['COMPLETION FOLLOW-UP PHASE'].isna())&\
                                (de_['COMPLETION FOLLOW-UP PHASE']!='YES')]
            
            target_df = pats_with_relapse_df.copy()
          
            
            idx_=list(set(compl_day_df.index) & set(target_df.index))
            
            target_df[['RELAPSE_DAY_','RELAPSE_']] = target_df[['RELAPSE_DAY','RELAPSE']].values
            
            ## Add real completion day of patients who had no relapse
            target_df.loc[idx_,'RELAPSE_DAY_']=compl_day_df.loc[idx_,'COMPLETION FOLLOW-UP PHASE'].astype(float).values
            target_df.loc[target_df['RELAPSE_DAY_']>10300,'RELAPSE_DAY']=np.nan
            #df.loc[df['RELAPSE_DAY']>1000,'RELAPSE_DAY']=1000
            
            
            ## For no relapse patients where completion day is missing, set scheduled follow-up as completion day (18 months for RemoxTB & 24 months for OFLOTUB)
            target_df.loc[(target_df['RELAPSE_DAY_'].isna())&\
                    (target_df['STUDYID']=='TB-1021'),'RELAPSE_DAY_']=550
            
            ## IN OFLOTUB, FOLLOW-UP WAS PERFORMED (SCHEDULED) TO 2 YEARS AFTER THERAPY COMPLETION
            target_df.loc[(target_df['RELAPSE_DAY_'].isna())&\
                    (target_df['ARM'].isin(['Control'])),'RELAPSE_DAY_']= (182 +  2 *365)
            
            target_df.loc[(target_df['RELAPSE_DAY_'].isna())&\
                    (target_df['ARM'].isin(['Gatifloxacin'])),'RELAPSE_DAY_']= (120 +  2 *365)


            target_df['therapy_arm_duration'] = '4-month'
            target_df.loc[target_df['ARM'].isin(['2EHRZ/4HR','Control']),'therapy_arm_duration'] = '6-month'

            mask= (target_df['DAYS_BETWEEEN_THERAPY_END_AND_RELAPSE'].isna()).values
            target_df.loc[mask,'DAYS_BETWEEEN_THERAPY_END_AND_RELAPSE'] = (target_df.loc[mask,'RELAPSE_DAY_'] - target_df.loc[mask,'last_therapy_day']).values
            
            target_df[['RELAPSE_DAY_SOT','RELAPSE_DAY_EOT']] = target_df[['RELAPSE_DAY_','DAYS_BETWEEEN_THERAPY_END_AND_RELAPSE']].values
            target_df[['RELAPSE_SOT','RELAPSE_EOT']] = target_df[['RELAPSE_','RELAPSE_']].values
            
            

            ## SET AMMAXIMAL DAY THRESHOLD
            # => Non-relapsing up until this threshold are getting censored at this timepoint
            #. => Pateints relapsing after this threshold are swithced to no No relapse, & are right-censored

        
            for time_origin_,max_day_ in zip(['SOT','EOT'],
                                            [730,365]):
                
                rel_day_coln=f'RELAPSE_DAY_{time_origin_}'
                rel_ind_coln=f'RELAPSE_{time_origin_}'
                
         
                
                #df_.loc[(~df_[rel_day_coln].isna())&\
                #        (df_[rel_day_coln]>max_day),rel_day_coln]=max_day
                        
                target_df.loc[(~target_df[rel_day_coln].isna())&\
                        (target_df[rel_day_coln]>max_day_),rel_ind_coln]=0
        
                #max_day=750
                target_df.loc[#(df['RELAPSE_DAY'].isna())&\
                        (target_df[rel_day_coln]>max_day_),rel_day_coln]=max_day_
                

            
            #target_df = target_df[['RELAPSE_','RELAPSE_DAY_']]
            target_df = target_df[[f'RELAPSE_{time_origin}',f'RELAPSE_DAY_{time_origin}']]

            target_df.columns=['RELAPSE','RELAPSE_DAY']
            pat_ids=target_df.index.tolist()
            y=target_df.copy()
            outcome_label_ = copy.deepcopy(outcome_label)

            
     

    if 'pred_prob' in parameters_for_analysis[data_param_key]['result_cat']:
        

        ## Create list to collect dataframes of different pred loss clusters
        clust_df_list=[]
        
        for model_name in model_names[1:]:
            print(model_name)

            ## DEFINE LABELS OF PRED. LOSS CLUSTERS CREATED IN S9_4_ML_on_LLM_embeddings.ipnyb!!!
            pred_loss_clust_labels={'raw_pred_prob_norm':{1.0:'hard_to_predict',2.0:'easy_to_predict'},                            
                                    'llm_pred_prob_norm':{1.0:'hard_to_predict',2.0:'easy_to_predict'},
                                    'raw_pred_prob':{2.0:'hard_to_predict',1.0:'easy_to_predict'},
                                    'llm_pred_prob':{1.0:'hard_to_predict',2.0:'easy_to_predict'}}
            
            ## Map the binary labels to 0 and 1
            label_2id_={'hard_to_predict':1,'easy_to_predict':0}
        
                
            for pred_prob_coln in ['raw_pred_prob_norm','raw_pred_prob',
                                   'llm_pred_prob_norm','llm_pred_prob',
                                   'diff_pred_prob_norm','diff_pred_prob'][:-2]:
                #print(pred_prob_coln)
                
                ## Load pred loss cluster
                if 'raw' in pred_prob_coln or 'diff' in pred_prob_coln:
                    data_dir_=f'../data/model_interpretation/baseline/pred_prob_clusters'
                    
                    training_data_type_='last_therapy_day'

                    fn=os.path.join(data_dir_,
                                f'tb21_22_2984_pats_22_vars_relapse_days_{training_data_type_}_{model_name}_{pred_prob_coln}_pred_prob_clusters.csv')
        
                if 'llm' in pred_prob_coln:
                    fn=f"../data/model_interpretation/LLM/pred_prob_clusters/tb21_22_2984_pats_22_vars_relapse_BioMistral-7B_base_LogisticRegression_full_all_days_autoenc_False_{pred_prob_coln}_pred_prob_clusters.csv"               
        
                try:
                    clust_df=pd.read_csv(fn,index_col=0)
                    
                except FileNotFoundError:
                    print(f'{fn} does not exist. Skipping to next model')
                    continue

                
                ## Map pred loss lcusters to binary (0:easy to predict; 1: hard to predict)
                clust_df_=clust_df.loc[clust_df['period']=='baseline',:]  
                clust_df_[f'{pred_prob_coln}_cluster_binary'] = clust_df_[f'{pred_prob_coln}_cluster'].map(pred_loss_clust_labels[pred_prob_coln]).map(label_2id_)
                clust_df_=clust_df_.sort_index()
                #clust_df_list.append(clust_df_[[f'{pred_prob_coln}_cluster_binary']])
                #print(clust_df_.columns)
                clust_df_list.append(clust_df_[[f'{pred_prob_coln}_cluster_binary',f'{pred_prob_coln}_loss'][:1]])

        ## Add all pred_prb_loss clusters to one dataframe
        clust_df_concat = pd.concat(clust_df_list,axis=1)

        ## Select final prediction label
        outcome_label_ = f'{outcome_label}_cluster_binary'
        
        clust_df_concat=clust_df_concat[~clust_df_concat[outcome_label_].isna()]
        
        target_df=clust_df_concat[[outcome_label_]]
        y=clust_df_concat[[outcome_label_]]
        pat_ids=clust_df_concat.index.tolist()
        
        
    return pat_ids,y,target_df,outcome_label_#,clust_df_concat


######====================================
## 1. Calibrate model with best hyperparams + extract calibrated prediction probs
## 2. Calculate confidence statistice from calirbated prediction probs.

def extract_confidence_and_ood_metrics(model,cv_roc_auc_scores,X_train,y_train,outcome_label,label_weights_dict):

    ## Calculate calibrated probabilities of trained best model
    from sklearn.calibration import CalibratedClassifierCV

    ## Extract train and validation ids to perform calibration on the same train-test sets as the hyperparam search
    cv_splits=zip(cv_roc_auc_scores['inner_train_val_splits']['train_ids'],
                  cv_roc_auc_scores['inner_train_val_splits']['test_ids'])
    
    sample_weights=y_train[outcome_label].map(label_weights_dict).values

    ## Fit calibrated model
    calibrated_model = CalibratedClassifierCV(estimator=model,
                                       cv=cv_splits,
                                       #n_jobs=-1,                                        
                                       method="isotonic")
    calibrated_model.fit(X_train.values, 
                           y_train.values,
                          sample_weight=sample_weights)

    ## Extract calibrated pred probs
    probs_calibrated = calibrated_model.predict_proba(X_train)[:, 1]
    probs_uncalibrated = model.predict_proba(X_train)[:, 1]


    
    """
    Compute per-model confidence statistics from probabilities.
    """
    eps = 1e-12
    p_cal = np.clip(probs_calibrated, eps, 1 - eps)
    p_uncal = np.clip(probs_uncalibrated, eps, 1 - eps)
    margin = np.abs(p_cal - 0.5)
    entropy = -(p_cal * np.log(p_cal) + (1 - p_cal) * np.log(1 - p_cal))
    conf_entropy = 1 - entropy / np.log(2)
    conf_metrics = pd.DataFrame({
        'prob_calibrated': p_cal,
        'prob_uncalibrated':p_uncal,
        'margin': margin,
        'entropy_conf': conf_entropy
    })
    if probs_uncalibrated is not None:
        conf_metrics['delta_calib'] = p_cal - p_uncal
    else:
        conf_metrics['delta_calib'] = 0.0

    conf_metrics.index=X_train.index

    ## Add out-of-distribution confidence variables (Mahalanobis and KNN confidence)
    ood_conf_df=cv_roc_auc_scores['ood_conf']
    conf_metrics.loc[ood_conf_df.index,ood_conf_df.columns.tolist()] = ood_conf_df.values
    
    return conf_metrics, calibrated_model





#### ======================
### PREDEICT BASELINE HAZARD WITH THE BRESLOW-METHOD ==> XGBOOSTT DOESN'T NATIVELY DO THAT,
def breslow_baseline_hazard(times, events, risk_scores):

    
    import numpy as np
    import pandas as pd
    #from xgboost import XGBRegressor
    
    """
    Estimate baseline cumulative hazard using Breslow estimator.
    
    times:       array of observed times
    events:      boolean array of event indicators  
    risk_scores: linear predictor from Cox model (NOT exponentiated)
    """
    times = np.array(times)
    events = np.array(events, dtype=bool)
    exp_scores = np.exp(risk_scores)
    
    # Only compute at event times
    event_times = np.sort(np.unique(times[events]))
    
    baseline_hazard = []
    
    for t in event_times:
        # Number of events at time t
        n_events_t = events[times == t].sum()
        
        # Sum of exp(risk) for all patients at risk at time t
        at_risk = times >= t
        sum_exp_risk = exp_scores[at_risk].sum()
        
        if sum_exp_risk > 0:
            baseline_hazard.append(n_events_t / sum_exp_risk)
        else:
            baseline_hazard.append(0.0)
    
    # Cumulative baseline hazard
    cumulative_baseline = np.cumsum(baseline_hazard)
    
    return event_times, cumulative_baseline


#### ======================
### PREDICT BASELINE HAZARD WITH THE BRESLOW-METHOD ==> XGBOOSTT DOESN'T NATIVELY DO THAT,
## COMPUTE SURVIVAL FUNCTION OF XGBOOST MODEL COMBINING THE BRESLOW HAZARD ESTIMATION METHOD WITH THE XGBOOST RISK OUTPUTS
def predict_survival_function(model, X_train, y_train_df, X_test):
    import numpy as np
    import pandas as pd
    from xgboost import XGBRegressor
    """
    Predict individual survival curves for test patients.
    Combines XGBoost risk scores with Breslow baseline hazard.
    
    Returns:
        event_times: timepoints of the baseline hazard
        survival_matrix: (n_test_patients x n_event_times) array
    """
    # Get risk scores for training data (for Breslow estimator)
    train_risk = model.predict(X_train)
    
    # Estimate baseline hazard from training data
    event_times, cum_baseline = breslow_baseline_hazard(
        times=y_train_df['RELAPSE_DAY'].values,
        events=y_train_df['RELAPSE'].astype(bool).values,
        risk_scores=train_risk
    )
    
    # Get risk scores for test patients
    test_risk = model.predict(X_test)
    
    # Individual survival: S(t|x) = exp(-H0(t) * exp(risk_score))
    # Shape: (n_test, n_event_times)
    survival_matrix = np.exp(
        -np.outer(np.exp(test_risk), cum_baseline)
    )
    
    return event_times, survival_matrix



'''
# Usage:
model = XGBRegressor(objective='survival:cox', 
                     eval_metric='cox-nloglik',
                     n_estimators=200, max_depth=3,
                     random_state=42, verbosity=0)

y_train_xgb = make_xgb_survival_labels(y_train_df)
model.fit(X_train, y_train_xgb)

event_times, surv_matrix = predict_survival_function(
    model, X_train, y_train_df, X_test
)

# surv_matrix[i, :] is the survival curve for test patient i
# Plot survival curve for patient 0:
import matplotlib.pyplot as plt
plt.step(event_times, surv_matrix[0, :], where='post')
plt.xlabel('Days from EOT')
plt.ylabel('Survival probability')
plt.show()
'''




## ==========================================
def check_ph_assumption_lifelines(sksurv_model, X_train, y_df_train):

    from lifelines import CoxPHFitter
    import pandas as pd
    import numpy as np
    from lifelines.statistics import proportional_hazard_test

    # Extract non-zero features from scikit-survival model
    if hasattr(sksurv_model, 'named_steps'):
        cox    = sksurv_model.named_steps['cox']
        scaler = sksurv_model.named_steps['scaler']
        #X_scaled = pd.DataFrame(
        #    scaler.transform(X_train),
        #    columns=X_train.columns
        #)
        X_scaled = X_train.copy()
    else:
        cox      = sksurv_model
        X_scaled = X_train.copy()

    # Get coefficients at selected alpha
    coef_matrix = cox.coef_
    if coef_matrix.ndim == 2:
        if hasattr(cox, 'alpha_'):
            best_idx = np.argmin(np.abs(cox.alphas_ - cox.alpha_))
        else:
            best_idx = coef_matrix.shape[1] // 2
        coef = coef_matrix[:, best_idx].ravel()
    else:
        coef = coef_matrix.ravel()

    # Keep only Lasso-selected features
    nonzero_idx      = np.where(coef != 0)[0]
    selected_features = X_train.columns[nonzero_idx].tolist()

    #print(f"Selected {len(selected_features)} non-zero features for PH test:")
    #print(selected_features)

    if len(selected_features) == 0:
        print("No non-zero features — cannot run PH test.")
        return None

    # Build fit dataframe with selected features only
    fit_df = X_scaled[selected_features].copy()
    fit_df['RELAPSE_DAY'] = y_df_train['RELAPSE_DAY'].values
    fit_df['RELAPSE']     = y_df_train['RELAPSE'].astype(int).values

    # Fit unpenalised Cox in lifelines on selected features only
    cph = CoxPHFitter(penalizer=0.01)
    cph.fit(
        fit_df,
        duration_col='RELAPSE_DAY',
        event_col='RELAPSE',
        show_progress=False
    )

    #cph.print_summary()
    results = proportional_hazard_test(cph, fit_df, time_transform='rank')
    #results.print_summary(decimals=3, model="untransformed variables")

    #print("\nChecking proportional hazards assumption:")
    #cph.check_assumptions(fit_df,p_value_threshold=0.05,show_plots=False)
    summary = results.summary
    summary = summary[['p']]

    return summary





'''


def check_ph_assumption(model, X_train, y_df_train, verbose=False,feature_names=None):
    from scipy.stats import spearmanr
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.pipeline import Pipeline

    # Handle both Pipeline and bare CoxnetSurvivalAnalysis
    if hasattr(model, 'named_steps'):
        cox     = model.named_steps['cox']
        scaler  = model.named_steps['scaler']
        X_scaled = scaler.transform(X_train)
        X_scaled = X_train.copy()#scaler.transform(X_train)
    else:
        # Bare model — assume X is already scaled or scale here
        cox      = model
        #X_scaled = X_train.values if hasattr(X_train, 'values') else X_train
        X_scaled = X_train.copy()

    # Convert to numpy immediately — prevents all pandas indexing errors
    X_scaled = np.array(X_scaled, dtype=float)
    
    # Rest of the function stays identical from here
    risk_scores = cox.predict(X_scaled)

    
    # Alternative: check what attributes are available
    #print([attr for attr in dir(cox) if 'alpha' in attr.lower()])
    
    # Most reliable fallback: use the last non-zero column
    # (CoxnetSurvivalAnalysis stores path from high to low regularisation)
    coef_matrix = cox.coef_  # (n_features, n_alphas)
    
    # Find column corresponding to selected alpha
    # The estimator sets self.alpha_ after fitting
    if hasattr(cox, 'alpha_'):
        best_alpha_idx = np.argmin(np.abs(cox.alphas_ - cox.alpha_))
        coef = coef_matrix[:, best_alpha_idx]
    elif hasattr(cox, 'best_coef_'):
        coef = cox.best_coef_.ravel()
    else:
        # Last resort: take column with most non-zeros that matches
        # the number of features in X_scaled
        coef = coef_matrix[:, -1].ravel()
    
    nonzero_idx = np.where(coef != 0)[0]
    print(f"coef shape: {coef.shape}, nonzero: {len(nonzero_idx)}")

    if feature_names is None:
        feature_names = [f'feature_{i}' for i in range(X_scaled.shape[1])]
    feature_names = np.array(feature_names)

    #print(f"Non-zero coefficients: {len(nonzero_idx)} / {len(coef)}")
    #print("Checking PH assumption for non-zero features only.\n")

    times  = y_df_train['RELAPSE_DAY'].astype(float).values
    events = y_df_train['RELAPSE'].astype(bool).values

    event_mask  = events == True
    event_times = times[event_mask]
    X_events    = X_scaled[event_mask]
    exp_scores  = np.exp(risk_scores)
    residuals   = []

    for i, t in enumerate(event_times):
        at_risk     = times >= t
        X_risk      = X_scaled[at_risk]
        exp_at_risk = exp_scores[at_risk]
        weights     = exp_at_risk / exp_at_risk.sum()
        expected_x  = (X_risk * weights[:, np.newaxis]).sum(axis=0)
        residuals.append(X_events[i] - expected_x)

    residuals = np.array(residuals)

    #print('residuals.shape',residuals.shape)
    #print('nonzero_idx.shape',nonzero_idx.shape)

    if verbose==True:
        print(f"{'Feature':<35} {'rho':>8} {'p-value':>10} {'Violation?':>12}")
        print("-" * 70)

    violations, bin_ind = [],[]
    for idx in nonzero_idx:
        rho, p    = spearmanr(event_times, residuals[:, idx])
        violation = p < 0.05
        #if violation:
        violations.append(feature_names[idx])
        if violation:
            bin_ind.append('y')
        if not violation:
            bin_ind.append('n')
        
        if verbose==True:
            flag = "YES *" if violation else "no"
            print(f"{feature_names[idx]:<35} {rho:>8.3f} {p:>10.4f} {flag:>12}")

    if verbose==True:
        print(f"\nViolations found: {len(violations)}")
        if violations:
            print("Violating features:", violations)
            print("→ Consider stratifying on these or noting as limitation.")
        else:
            print("→ PH assumption holds for all non-zero features.")
    if verbose==True:
        if violations:
            from scipy.ndimage import uniform_filter1d
            n_plots = len(violations)
            ncols=4
            nrows=int(np.ceil(n_plots/ncols))
            fig, axes = plt.subplots(nrows, ncols, 
                                      figsize=( ncols*3, nrows* 3), 
                                      )
            axes = axes.flatten()
            for ax, feat_name in zip(axes, violations):
                idx = list(feature_names).index(feat_name)
                ax.scatter(event_times, residuals[:, idx], alpha=0.4, s=15)
                order  = np.argsort(event_times)
                smooth = uniform_filter1d(residuals[order, idx], size=10)
                ax.plot(event_times[order], smooth, color='red', linewidth=2)
                ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
                ax.set_xlabel('Event time (days)')
                ax.set_ylabel('Schoenfeld residual')
                ax.set_title(f'{feat_name}\n(PH violated)')
    
            plt.tight_layout()
            plt.show()

    violations_ = pd.Series(index=violations,
                             data=bin_ind).to_frame().T

    return violations_
'''    
