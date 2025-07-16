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
    
    X_subset_=X_subset.copy()

    #print('before backfill func',X_subset_.loc[X_subset_['USUBJID']=='TB-1022/53003',['DAY']+final_cols_].sort_values('DAY'))


    ## Loop over patients who have missing data in their early visits, and backward fill missing data
    for pat in pats_with_miss_vars[:]:
        #print(pat)
        pat_df=X_subset[X_subset['USUBJID']==pat]
            
    
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
                                                                                     
def create_std_training_testing_data(X,y,pat_ids_,test_size_ratio,rand_state,training_data_type,columns_to_drop,period_end_day,outcome_label):


    
    ## STRATIFY ON OUTCOME LABEL & STUDYID 
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




##=========================================
def run_cv(X,y,k_folds,model_name,weight_by_label_freq,random_state,outcome_label,model_params,train_params):
    
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
    
    cv_roc_auc_scores={}

    ## STRATIFY ON OUTCOME LABEL & STUDYID 
    ##. ==> WITHIN STUDY ROC-AUC CLAUCLATION IS POSSIBLE, AS THERE ALWAYS WILL BE AT LEAST ONE UNFAVOUR. LABEL FROM BOTH STUDIES IN THE TEST SET
    df_=y.reset_index().drop_duplicates(subset='USUBJID')#.set_index('USUBJID',drop=True)
    df_['STUDYID']=df_['USUBJID'].str.split('/',expand=True)[0].values
    df_=df_.set_index('USUBJID')
    y_for_strat=df_[outcome_label].astype(str) + "_" + df_['STUDYID']#.astype(str)

    pat_ids=df_.index
    n_of_classes=len(y[outcome_label].unique())

    # Loop through each fold
    #for train_index, test_index in skf.split(pat_ids, y_unique.loc[pat_ids,outcome_label]):
    for train_index, test_index in skf.split(pat_ids, y_for_strat.loc[pat_ids]):
        
        train_pat_ids=pat_ids[train_index]
        test_pat_ids=pat_ids[test_index]
        
        #print('train test index set',set(train_pat_ids)&set(test_pat_ids))
        
        train_mask=y.index.get_level_values('USUBJID').isin(train_pat_ids)
        test_mask=y.index.get_level_values('USUBJID').isin(test_pat_ids)
        
        
        # Split the data and standardise them by the training data's distribution
        #X_train_fold, X_test_fold = scale_by_training_data(X.loc[train_mask,:], X.loc[test_mask,:])
        X_train_fold, X_test_fold = X.loc[train_mask,:], X.loc[test_mask,:]
        y_train_fold,y_test_fold=y.loc[train_mask,:],y.loc[test_mask,:]

        
        ## Init model
        model,label_weights,_=init_model(model_name,X_train_fold,y_train_fold,
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





##=========================================
## RUN CV OF GIVEN MODEL & TRAIN FINAL MODEL AFTERWARDS
def run_parameter_search(model_name,X_train,y_train_data,
                                k_folds,random_state,
                                outcome_label,
                                param_search_dict,
                                weight_by_label_freq,
                                train_params):
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
  
        #print(f'Running CV of {model_name} model with parameters:{param_set_string}')
        random_state_=random_state #+n
              
        ## CALCULATE CV-SCORES
        cv_roc_auc_scores=run_cv(X_train,y_train_data,k_folds,
                                 model_name,weight_by_label_freq,
                                 random_state_,outcome_label,
                                 model_params,
                                train_params)
        
        ## SAVE RESULT OF CV WITH GIVEN PARAMETER SET
        param_search_results[param_set_string]=cv_roc_auc_scores

        
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


##=========================================
## RUN CV OF GIVEN MODEL & TRAIN FINAL MODEL AFTERWARDS
def calc_roc_auc_score_of_model(model_name,X_train,y_train_data,
                                k_folds,random_state,
                                outcome_label,
                                model_params,
                                weight_by_label_freq,
                               train_params):
        
    ### RUN CV & TRAIN MODEL AFTERWARDS
    if model_name in ['XGBoost','GradientBoost']:
        
        ## CALCULATE CV-SCORES
        cv_roc_auc_scores=run_cv(X_train,y_train_data,k_folds,
                                 model_name,weight_by_label_freq,
                                 random_state,outcome_label,
                                 model_params,
                                 train_params)
        
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
                                 train_params)
        
        ## INITIALIZE MODEL & TRAIN 
        model,label_weights,label_weights_dict=init_model(model_name,X_train,y_train_data,
                                                          k_folds,random_state,outcome_label,
                                                          model_params,
                                                          weight_by_label_freq,
                                                          train_params) 

        #X_train,_ = scale_by_training_data(X_train, X_train)
        model.fit(X_train, y_train_data)
        
    return model,cv_roc_auc_scores,label_weights_dict



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

def extract_21_22_relapse_pats():

    ### ====== LOAD ALL PATIENTS DATA =======
    fn='../data/tb21_22_2984_pats_22_vars_result_at_end_of_treatment_preproc_data_with_imp.csv.gz'
    X_subset=pd.read_csv(fn,index_col=0)
    X_subset=X_subset.rename(columns=lambda x: x.replace('<', 'lower than'))
    X_subset=X_subset.rename(columns=lambda x: x.replace('>', 'higher than'))
    
    
    #### ================================ FAVOURABLE PATIENTS ========== ############
    
    ### COLLECT PATIENTS, WHO ONLY HAVE FAVOURABLE OUTCOMES AT END OF TREATMENT & AT ALL FOLLOW-UP TIMEPOINTS 
    #. ==>TB-1021: 12 & 18 MONTHS, TB-1022: 18 & 24 MONTHS

   
    
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

    pats_relapse_df['DAYS_BETWEEEN_THERAPY_END_AND_RELAPSE']=(pats_relapse_df['RELAPSE_DAY'] - pats_relapse_df['last_therapy_day']).values

    return pats_relapse_df#,relapse_during_obs_period,relapse_after_obs_period,pats_with_sparse_relapse_data,max_days

##========================================= 
def extract_last_init_therapy_day_from_drug_regimen(pat_ids):

    ## Load drug regimen data, containing the daily taken doses
    dr_reg=pd.read_csv('../data/out_temporal_pat_regimens_1018_20_21_22_30.csv.gz',index_col=0)
    dr_reg=dr_reg[dr_reg['USUBJID'].isin(pat_ids)]

    ## Extract colnames containing the drug names
    drug_cols=[c.split('_cumulative_dose')[0] for c in dr_reg.columns if '_cumulative_dose' in c]

    ## IF drug was not taken anymore, the daily dose is set to 0. Drop all drug columns which have only zeroes (meaning patient didn't take them),
    #. replace 0s with NaNs, and drop all rows, which only have NaNs in the drug column 
    #. ==> The last day where drug was applied is the last day of initial therapy
    last_init_therapy_days=dr_reg.loc[:,['DAY','USUBJID']+drug_cols].groupby('USUBJID').apply(lambda x:x.replace(0,np.nan).dropna(subset=drug_cols,how='all',axis=0)['DAY'].max())

    
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
        last_init_day_of_pats_with_retreatment=dr_reg.loc[dr_reg['USUBJID'].isin(pats_with_treatment_gaps), ['DAY', 'USUBJID']].groupby('USUBJID', as_index=True).apply(
                                                            lambda x: x.loc[x['DAY'] < retreat_start_day.loc[x['USUBJID'].unique()[0]].values[0], 'DAY'].max()
                                                            ).sort_values(ascending=False)
    
        
        ## Replace the last day values of patients with treatment gaps with the extracted last day values
        last_init_therapy_days.loc[last_init_day_of_pats_with_retreatment.index]=last_init_day_of_pats_with_retreatment.values

    return last_init_therapy_days

##========================================= 
def subset_pats_with_therapy_in_period(period_num,period_end_days,last_init_ther_days,
                                       pats_with_relapse_df,period_end_day,data_param_key):
    
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
    if parameters_for_analysis[data_param_key]['result_cat']=='RELAPSE': 
        
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

    ## One-hot encode 'ZN-smear oridinal column'
    #if 'mb_ZN-smear_STD_CAT_ORDINAL_RESULT' in X.columns:
    #    X = pd.concat([X.drop(columns=['mb_ZN-smear_STD_CAT_ORDINAL_RESULT']),\
    #                   pd.get_dummies(X['mb_ZN-smear_STD_CAT_ORDINAL_RESULT'].astype('Int64'),dtype=int,prefix='mb_ZN-smear')],axis=1)
    X=X.drop(columns=['mb_ZN-smear_STD_RESULT'])
    
    X['vs_BMI_STD_NUM_RESULT'] = (X['vs_Weight_STD_NUM_RESULT']/(X['vs_Height_STD_NUM_RESULT']/100)**2).values
    X['ARM']=X['ARM'].replace({'Gati-arm regimen (4 month regimen)':'Gatifloxacin (4 month)',
                             'Control-arm regimen (6 month regimen)':'Control (6 month)'},regex=False)

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
    

    '''
    ## SPLIT THE CUMULATIVE DOSES OF THE PATIENTS BETWEEN THE ARMS ==> CONTROLLING FOR THE DIFFERENT NUMBER OF SCHEDULED DOSES BETWEEN ARMS
    arm_cumul_colnames=[f"{arm}_drugs_cumul" for arm in X['ARM'].unique()]
    X[arm_cumul_colnames]=0
    
    for arm in X['ARM'].unique():
        X.loc[X['ARM']==arm,f"{arm}_drugs_cumul"]=X.loc[X["ARM"]==arm,'dr_reg_study_drugs_cumul'].values
    
    X=X.drop(columns=['dr_reg_study_drugs_cumul'])
    '''

    return X,race_colnames
