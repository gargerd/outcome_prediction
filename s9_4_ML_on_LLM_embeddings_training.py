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
import scanpy as sc
import time
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset,DataLoader
import copy
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300

from s9_4_ML_on_LLM_embeddings_functions import *



### EXTRACT PARAMETERS FOR PARAMETER SEARCH FROM ARGPARSE
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", help="One of the dataset names from the keys of parameters_for_analysis dictionar (see below)")
parser.add_argument("--data_inclusion_type", help="One of the 3 data inclusion types: ['baseline_last_day','baseline_vars','all_days']")
parser.add_argument("--model", help="One of the 2 models: ['XGBoost','LogisticRegression']")
parser.add_argument("--period_end_day", help="One of the 6 periods: [baseline,31,62,93,125,160,'all']")


args = parser.parse_args()
dataset_name_=[args.dataset_name]
data_inclusion_type_ = [args.data_inclusion_type]
model_names_ = [args.model]
period_end_day_=[args.period_end_day if args.period_end_day in ['baseline','all'] else int(args.period_end_day)]

'''
if args.cpu_cores is not None:
    cpu_cores_ = range(int(args.cpu_cores.split('-')[0]),int(args.cpu_cores.split('-')[1])+1)
    
    ## Extract the ids of CPU cores available & subset them to the specified number
    avail_cores=list(os.sched_getaffinity(0))
    cores_to_run_on = [avail_cores[n] for n in cpu_cores_]
    
    #import os
    os.sched_setaffinity(0, cores_to_run_on) 
'''
print('data_inclusion_type_',data_inclusion_type_,'model_names_',model_names_,'period_end_day_',period_end_day_)


# DEFINE DATASET NAMES + FUNCTIONS
parameters_for_analysis={'tb21_22_2984_pats_22_vars_result_at_end_of_treatment':{
                            'fn':'tb21_22_2984_pats_22_vars_result_at_end_of_treatment',
                            'result_cat':'RESULT_AT_END_OF_TREATMENT',
                            'selection_method':'patient_clustering',
                            'clust_comb':'2-3-4-5',
                            'graph_metric':None,
                            'num_of_common_vars':22,
                            'training_days':120},
            

                        'tb21_22_2984_pats_22_vars_relapse':{
                            'fn':'tb21_22_2984_pats_22_vars_result_at_end_of_treatment',
                            'result_cat':'RELAPSE',
                            'selection_method':'patient_clustering',
                            'clust_comb':'2-3-4-5',
                            'graph_metric':None,
                            'num_of_common_vars':22,
                            'training_days':120}, 

                         'tb21_22_2984_pats_22_vars_relapse_without_dr_reg':{
                            'fn':'tb21_22_2984_pats_22_vars_relapse_without_dr_reg',#_wo_dr_reg',
                            'X_fn':'tb21_22_2984_pats_22_vars_result_at_end_of_treatment',
                            'pat_ids_fn':'tb21_22_2984_pats_22_vars_relapse',
                            'result_cat':'RELAPSE'},

                           'tb21_22_2984_pats_22_vars_raw_pred_prob_norm_loss':{
                            'fn':'tb21_22_2984_pats_22_vars_result_at_end_of_treatment',
                            'result_cat':'raw_pred_prob_norm'},
                         
                        'tb21_22_2984_pats_22_vars_llm_pred_prob_norm_loss':{
                            'fn':'tb21_22_2984_pats_22_vars_result_at_end_of_treatment',
                            'result_cat':'llm_pred_prob_norm'},
                         
                         
                         'tb21_22_2840_pats_23_vars_result_at_end_of_treatment':{
                            'fn':'tb21_22_2840_pats_23_vars_result_at_end_of_treatment',
                            'result_cat':'RESULT_AT_END_OF_TREATMENT',
                            'selection_method':'patient_clustering',
                            'clust_comb':'2-3-4-5',
                            'graph_metric':None,
                            'num_of_common_vars':23,
                            'training_days':120},

                         'tb21_22_2840_pats_23_vars_relapse':{
                            'fn':'tb21_22_2840_pats_23_vars_result_at_end_of_treatment',
                            'result_cat':'RELAPSE',
                            'selection_method':'patient_clustering',
                            'clust_comb':'2-3-4-5',
                            'graph_metric':None,
                            'num_of_common_vars':23,
                            'training_days':120},
    
    

                         'tb21_22_2798_pats_24_vars_result_at_end_of_treatment':{
                            'fn':'tb21_22_2798_pats_24_vars_result_at_end_of_treatment',
                            'result_cat':'RESULT_AT_END_OF_TREATMENT',
                            'selection_method':'patient_clustering',
                            'clust_comb':'2-3-4-5',
                            'graph_metric':None,
                            'num_of_common_vars':24,
                            'training_days':120},   

                          'tb21_22_2798_pats_24_vars_relapse':{
                            'fn':'tb21_22_2798_pats_24_vars_result_at_end_of_treatment',
                            'result_cat':'RELAPSE',
                            'selection_method':'patient_clustering',
                            'clust_comb':'2-3-4-5',
                            'graph_metric':None,
                            'num_of_common_vars':24,
                            'training_days':120}, 


                         'tb21_1405_pats_40_vars_result_at_end_of_treatment':{
                            'fn':'tb21_1405_pats_40_vars_result_at_end_of_treatment',
                            'result_cat':'RESULT_AT_END_OF_TREATMENT',
                            'selection_method':'patient_clustering',
                            'clust_comb':'1-3-4-5',
                            'graph_metric':None,
                            'num_of_common_vars':40,
                            'training_days':120},
                         
                         'tb21_1405_pats_40_vars_relapse':{
                            'fn':'tb21_1405_pats_40_vars_result_at_end_of_treatment',
                            'result_cat':'RELAPSE',
                            'selection_method':'patient_clustering',
                            'clust_comb':'1-3-4-5',
                            'graph_metric':None,
                            'num_of_common_vars':40,
                            'training_days':120},
                         
                         }


param_search_dict={'RandomForest':{'n_estimators':[300,500,700],
                                   'max_features':['sqrt'],
                                   'max_depth':[3,5,7,9]},
                   
                  'GradientBoost':{'n_estimators':[300,500,700],
                                   'max_features':['sqrt'],
                                   'learning_rate':[0.1,0.3,0.5,0.8]},
                  
                  'XGBoost':{'n_estimators':[300,500,700],
                             'max_depth':[3,5,7,9],
                             'eta':[0.1,0.3,0.5,0.8],
                             #'subsample':[1.0,0.9,0.8,0.7]
                             #'tree_method':['hist'],
                             #"device": ["cuda"]
                             #'n_jobs':[4]
                            },
                   
                   'LogisticRegression':{#'l1_ratio':[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
                                       'n_jobs':[4],
                                       'l1_ratio':[1,0.5,0.1],
                                        'C': [0.001, 0.01, 0.1, 1, 10]},
                   
                  'SVC':{'C':[1e-3,1e-2,1e-1,1e0,1e1,1e2],
                        'kernel':['rbf','poly']},
                   
                  'KNN':{'n_neighbors':[2,5,10,25,50,100]},
                   
                  'Dense':{'learning_rate':[1e-2],#1e-3],
                          'batch_size':[32],
                          'num_epochs':[200],},
                  }  



## Define string descriptions to add as a prefix for each ds_type, for LLM to know what the variables describe
ds_type_descriptions={'dm':'Demographic descriptors',
                    'mb':'Microbiological test results',
                    'vs':'Vital signs',
                    're':'Chest X-ray findings',
                    'lb':'Laboratory test results',
                    'dr_reg':'Cumulative drug doses taken'}


###### LOAD LLM EMBEDDINGS

import warnings
embed_dict={}



llm_model_names=['BioMistral/BioMistral-7B',"epfl-llm/meditron-70b","epfl-llm/meditron-7b",
                 "google/long-t5-local-base",'google/long-t5-tglobal-large',
                'text-embedding-3-small']

fine_tuned_tags=['base','fine_tuned']

data_inclusion_types=['baseline_last_day','baseline_vars','all_days'][:]


## Last days of periods to use as training data.
# i.e period_end_day=31 ==> use only data of patient coming from the first 30 days
period_end_days=['baseline',31,62,93,125,160,'all']
autoenc_merged_bool_list=[False,True]

'''
for data_param_key in dataset_name_:
    
    embed_dict[data_param_key]={}
    outcome_label=data_param_key.split('vars_')[-1].upper()
    
    for llm_model_name in llm_model_names[:1]:

        for fine_tuned_tag in fine_tuned_tags[:1]:
            
            if 'text-embedding' in llm_model_name:
                llm_model_name_with_tag=llm_model_name
            else:
                llm_model_name_with_tag='_'.join([llm_model_name.split('/')[-1],fine_tuned_tag])
            
            embed_dict[data_param_key][llm_model_name_with_tag]={}
            
            for period_end_day in period_end_day_:
                
                embed_dict[data_param_key][llm_model_name_with_tag][period_end_day]={}
                
                for data_inclusion_type in data_inclusion_type_:

                    embed_dict[data_param_key][llm_model_name_with_tag][period_end_day][data_inclusion_type]={}
                    
                    for autoencoder_merged in autoenc_merged_bool_list[:1]:

                        ## Load embeddings created by LLM model
                        print(f'Loading embeddings of {data_param_key} data / {llm_model_name_with_tag} /{period_end_day} /{data_inclusion_type} / autoencoder {autoencoder_merged} model')
                        #try:
                        df_pt=load_input_emebddings_of_model(data_param_key,llm_model_name,fine_tuned_tag,period_end_day,data_inclusion_type,autoencoder_merged)
                        #except FileNotFoundError:
                        #    warnings.warn(f'Embeddings of {data_param_key} data / {llm_model_name_with_tag} model not found! Check if they have been created. Skipping to next item.')
                        #    continue
                        print('Loaded embeddings with shape',df_pt.shape)
                        embed_dict[data_param_key][llm_model_name_with_tag][period_end_day][data_inclusion_type][autoencoder_merged]=df_pt
        #            embed_dict[data_param_key][llm_model_name_with_tag]['pca_mean_vect']=df_npy

'''
import warnings
import time
from tqdm import tqdm

embed_dict = {}

llm_model_names = ['BioMistral/BioMistral-7B', "epfl-llm/meditron-70b", "epfl-llm/meditron-7b",
                   "google/long-t5-local-base", 'google/long-t5-tglobal-large',
                   'text-embedding-3-small']
fine_tuned_tags = ['base', 'fine_tuned']
data_inclusion_types = ['baseline_last_day', 'baseline_vars', 'all_days']
period_end_days = ['baseline', 31, 62, 93, 125, 160, 'all']
autoenc_merged_bool_list = [False, True]

# Prepare all combinations to wrap in tqdm
combinations = []
for data_param_key in dataset_name_:
    #outcome_label = data_param_key.split('vars_')[-1].upper()
    outcome_label = parameters_for_analysis[data_param_key]['result_cat']
    
    for llm_model_name in llm_model_names[:1]:
        
        for fine_tuned_tag in fine_tuned_tags[:1]:
            
            if 'text-embedding' in llm_model_name:
                llm_model_name_with_tag = llm_model_name
            else:
                llm_model_name_with_tag = '_'.join([llm_model_name.split('/')[-1], fine_tuned_tag])
                
            for period_end_day in period_end_day_:
                
                for data_inclusion_type in data_inclusion_type_:
                    
                    for autoencoder_merged in autoenc_merged_bool_list[:1]:
                        combinations.append((
                            data_param_key, llm_model_name, fine_tuned_tag,
                            llm_model_name_with_tag, period_end_day,
                            data_inclusion_type, autoencoder_merged
                        ))

# Iterate with progress bar
for combo in tqdm(combinations, desc="Loading embeddings", unit="set"):
    data_param_key, llm_model_name, fine_tuned_tag, llm_model_name_with_tag, period_end_day, data_inclusion_type, autoencoder_merged = combo
    
    embed_dict.setdefault(data_param_key, {})
    embed_dict[data_param_key].setdefault(llm_model_name_with_tag, {})
    embed_dict[data_param_key][llm_model_name_with_tag].setdefault(period_end_day, {})
    embed_dict[data_param_key][llm_model_name_with_tag][period_end_day].setdefault(data_inclusion_type, {})

    print(f"\n→ Loading: {data_param_key} | {llm_model_name_with_tag} | {period_end_day} | {data_inclusion_type} | autoenc={autoencoder_merged}")

    try:
        start = time.time()
        df_pt = load_input_embeddings_of_model(
            data_param_key, llm_model_name, fine_tuned_tag,
            period_end_day, llm_model_name_with_tag,data_inclusion_type, autoencoder_merged
        )
        elapsed = time.time() - start
        print(f"Loaded in {elapsed:.2f} seconds.")
    except FileNotFoundError:
        warnings.warn(f"Embeddings missing for: {data_param_key} / {llm_model_name_with_tag}. Skipping.")
        continue

    embed_dict[data_param_key][llm_model_name_with_tag][period_end_day][data_inclusion_type][autoencoder_merged] = df_pt
    #print('Inf present:',any(df_pt==np.inf))
    #print('-Inf present:',any(df_pt==-np.inf))
    assert not np.isinf(df_pt.values).any(), "There are still Inf values!"
    assert not np.isnan(df_pt.values).any(), "There are still NaN values!"
    

####=================================================
## Define training parameters
from sklearn.model_selection import train_test_split
import warnings
from tqdm import tqdm


outcome_df=pd.read_csv('../data/tb_1018_20_21_22_30_outcome.csv.gz',index_col=0)
outcome_df=outcome_df.set_index('USUBJID',drop=True)

llm_model_names=['BioMistral/BioMistral-7B',"epfl-llm/meditron-70b","epfl-llm/meditron-7b",
                 "google/long-t5-local-base",'google/long-t5-tglobal-large',
                'text-embedding-3-small']

fine_tuned_tags=['base','fine_tuned']
model_names=['XGBoost','GradientBoost','LogisticRegression','RandomForest']#,'SVC','KNN']
model_names=['XGBoost','LogisticRegression']#,'Dense']#,'SVC','KNN']

#model_names=['LogisticRegression','XGBoost'][:]
training_data_types=['full','pca']
#data_inclusion_types=['baseline_vars','baseline_vars_ext','all_days'][:]
#data_inclusion_types=['baseline_last_day','baseline_vars','all_days'][:]

## Set up prediction labels
id2label={0: "FAVOURABLE", 1: "UNFAVOURABLE"}
label2id={"FAVOURABLE": 0, "UNFAVOURABLE": 1}


## Drop patients who have their last data at an earlier timepin than threshold
therapy_day_thr=80
period_end_days=['baseline',31,62,93,125,160,'all']

## Define training parameters
train_params={'num_cv_repeats':25,
              'k_folds':5,              
              'weight_by_label_freq':True,
              'random_state':42,
              'test_size_ratio':0.2,
              'pca_comp':512,
              'label_weights':[1,1], ## [index_0: weight for label 0 (negative),index_1: weight for label 1 (positive)], only
                                     ## only considered if weight_by_label_freq=False !
              'label2id':label2id}


dense_network_params={#'input_dim' :len(data.drop(columns=['STUDYID','ARM']).columns),
                        'batch_size':128,
                        'hidden_dims':[128,32],
                        'output_dim':len(id2label.keys()),
                        'learning_rate':1e-4,
                        'weight_decay':1e-5,
                        'dropout_prob':0.1,
                        'num_epochs':5,
                        'label_weights':[1,30],
                        'hidden_activation':nn.ReLU(), #nn. Sigmoid()
                        'last_activation': nn.Softmax(dim=1), # #nn.Hardsigmoid()
                        'criterion':'CrossEntropyLoss'
                        }



####=================================================
## Run parameter search for models
import warnings
warnings.filterwarnings("ignore")


import time    
start=time.time()

### loop over ML models, train them & training results in a dictionary
for data_param_key in dataset_name_:

    ## LOAD FINAL PATIENT IDS FOR ANALYSIS, SAVED DURING PREPROCESSING OF THE BASELINE MODELS IN NOTEBOOK S9_3
    if 'pat_ids_fn' in parameters_for_analysis[data_param_key].keys():
        fn=f"../data/{parameters_for_analysis[data_param_key]['pat_ids_fn']}_final_pat_ids_for_analysis.pickle"
    else:  
        fn=f'../data/{data_param_key}_final_pat_ids_for_analysis.pickle'
    with open(fn, 'rb') as handle:
        final_pat_ids_for_analysis=pickle.load(handle)
    
    print(data_param_key)
    #fn='../data/'+data_param_key+'_all_data_concat.csv.gz'
    #X=pd.read_csv(fn,index_col=0,low_memory=False)
    
    #outcome_label=data_param_key.split('vars_')[-1].upper()
    outcome_label = parameters_for_analysis[data_param_key]['result_cat']
        
    for llm_model_name in llm_model_names[:1]:
        
        for fine_tuned_tag in fine_tuned_tags[:1]:
            

            if 'text-embedding' in llm_model_name:
                llm_model_name_with_tag=llm_model_name
            else:
                llm_model_name_with_tag='_'.join([llm_model_name.split('/')[-1],fine_tuned_tag])
            
            
            for period_end_day in period_end_day_:                
                period_num = period_end_days.index(period_end_day)
                
                print(f'Training on {period_end_day} days of data')

                for data_inclusion_type in data_inclusion_type_:

                    for autoencoder_merged in autoenc_merged_bool_list[:1]:

                        ## Load embeddings created by LLM model
                        print(f'Loading embeddings of {data_param_key} data / {llm_model_name_with_tag} /{period_end_day} /{data_inclusion_type} /autoenc {autoencoder_merged} model')
    
                        ## Load embeddings created by LLM model
                        #print(f'Loading embeddings of {llm_model_name_with_tag} model')
                        try:
                            df=embed_dict[data_param_key][llm_model_name_with_tag][period_end_day][data_inclusion_type][autoencoder_merged]
                        except KeyError:
                            warnings.warn(f"Embeddings of {parameters_for_analysis[data_param_key]['fn']} data / {llm_model_name_with_tag} model/ {period_end_day} days/ {data_inclusion_type}/ autoenc {autoencoder_merged}not found! Check if they have been created. Skipping to next item.")
                            continue
        
                        #print(data_param_key)
                        if 'X_fn' in parameters_for_analysis[data_param_key].keys():
                            fn='../data/'+parameters_for_analysis[data_param_key]['X_fn']+'_all_data_concat.csv.gz'
                        else:
                            fn='../data/'+parameters_for_analysis[data_param_key]['fn']+'_all_data_concat.csv.gz'
                        X=pd.read_csv(fn,index_col=0,low_memory=False)

                        ## Return dataframe with the outcome label
                        pat_ids,y,target_df,outcome_label = return_predict_label_dataframe(parameters_for_analysis,data_param_key,X,
                                                                                      outcome_df,outcome_label,model_names)
                       
                        ## Drop patients who have their last therapy day before therapy_day_thr ==> these patient probably dropped out
                        #last_day_per_pat_df=X.sort_values(by=['DAY']).groupby('USUBJID').apply(lambda x: x.loc[x.index[-1],:])
                        #pat_ids=last_day_per_pat_df[last_day_per_pat_df['DAY']>therapy_day_thr]['USUBJID'].tolist()

                        ## Subset initial therapy last day dataframe to all patient considered in analysis
                        #init_ther_df=last_initial_therapy_day_df.loc[pat_ids,:]
                        last_init_ther_days = extract_last_init_therapy_day_from_drug_regimen(pat_ids)

                        ## SUBSET TO PATIENTS WHO WERE TAKING DRUGS DURING THE PERIOD
                        #pat_ids_ = subset_pats_with_therapy_in_period(period_num,period_end_days,data_param_key,last_init_ther_days)
                        #pat_ids_= final_pat_ids_for_analysis[period_end_day]['X_train_ids'] + final_pat_ids_for_analysis[period_end_day]['X_test_ids']
                        
                       # df=df.loc[list(set(df.index) & set(pat_ids_)),:]
                        #df=df.loc[(pat_ids_),:]
        
        
                        ## Standardise data + calculate PCA 
                        #scaled_data,df_pca=return_std_data_and_pca(df,train_params['pca_comp'])
                        #df_pca=(scaled_data.loc[:,sign_diff_cols['Column'][:20]])
                        df_pca = return_pca(df,train_params['pca_comp'])
                        X_dict={'full':df,'pca':df_pca}
        
        
                        df_=y.reset_index()#
                        df_['STUDYID']=df_['USUBJID'].str.split('/',expand=True)[0].values
                        #print(pd.crosstab(df_.loc[df_['USUBJID'].isin(pat_ids_),'STUDYID'],df_.loc[df_['USUBJID'].isin(pat_ids_),outcome_label]))
                        
                        for training_data_type in training_data_types[:1]:
        
        
                            ## Define X and y dataframes for training
                            X=X_dict[training_data_type].copy()
                            #y=y.loc[y.index.get_level_values('USUBJID').isin(pat_ids_)]
                            #y=y.loc[pat_ids_]
                            
                            
                            print(f'+++++++\nRunning training with {training_data_type} model: {X.shape[1]} vars\n+++++++')
        
                            #for model_name in model_names[:]:
                            for n, model_name in enumerate(tqdm(model_names_, desc="Processing", unit="model")):
                                print('\n=====================')
                                print(model_name)
        
                                training_results={}
                                training_results['train_params']=train_params
                                training_results['cv_results']={}

                                ## Take the mean of the best parameteres selected for each CV-split as best parameters for the final model
                                metric_func=np.mean
                                #num_of_top_models_per_cv=5
                                num_of_top_models_per_cv=1

                                if model_name=='XGBoost':

                                    ## LOAD RESULTS OF PARAMETER SEARCH AND EXTRACT THE PARAMETERS OF THE BEST MODEL
                                    fn=f'../data/{data_param_key}_{llm_model_name_with_tag}_{model_name}_{period_end_day}_days_{training_data_type}_{X.shape[1]}_{data_inclusion_type}_autoenc_{autoencoder_merged}_vars_param_search_results.pickle'
                                    with open(fn, 'rb') as handle:
                                        param_search_results=pickle.load(handle)
                                    
                                    best_model_params_across_cvs=extract_best_model_params(param_search_results,metric_func,num_of_top_models_per_cv,average_models_across_splits=False)

                                if model_name=='LogisticRegression':
                                    num_of_top_models_per_cv = min(1,len(param_search_dict[model_name]['l1_ratio']))
                                
                                for cv_repeat_num in tqdm(range(train_params['num_cv_repeats']),desc="Processing", unit="cv_repeat"):
                                    rand_state=train_params['random_state'] + cv_repeat_num

                                    '''
                                    ## STRATIFY ON OUTCOME LABEL & STUDYID 
                                    ##. ==> WITHIN STUDY ROC-AUC CLAUCLATION IS POSSIBLE, AS THERE ALWAYS WILL BE AT LEAST ONE UNFAVOUR. LABEL FROM BOTH STUDIES IN THE TEST SET
                                    y_for_strat=y[outcome_label].astype(str) + "_" + y.index.get_level_values('STUDYID')#.astype(str)
                                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=train_params['test_size_ratio'], 
                                                                                        stratify=y_for_strat,random_state=rand_state)
                                    '''
                                    X_train_ids=final_pat_ids_for_analysis[period_end_day][cv_repeat_num]['X_train_ids']
                                    X_test_ids=final_pat_ids_for_analysis[period_end_day][cv_repeat_num]['X_test_ids']
                                    
                                    X_train=X.loc[list(set(X.index)&set(X_train_ids)),:]
                                    X_test=X.loc[list(set(X.index)&set(X_test_ids)),:]

                                    y_train=y.loc[list(set(X.index)&set(X_train_ids)),:]
                                    y_test=y.loc[list(set(X.index)&set(X_test_ids)),:]
                                    
                                    #if model_name=='LogisticRegression':
                                    X_train, X_test = scale_by_training_data(X_train, X_test)

                                    
                                    ## SAVE PARAMETERRS OF CV-SPLIT
                                    print(f"Num of CV-repeat:{cv_repeat_num+1}")
                                    training_results['cv_results'][f'cv_rep_{cv_repeat_num}']={}

                                    training_results['cv_results'][f'cv_rep_{cv_repeat_num}']['X_train_indices']=X_train.index.tolist()
                                    training_results['cv_results'][f'cv_rep_{cv_repeat_num}']['X_test_indices']=X_test.index.tolist()
                                    training_results['cv_results'][f'cv_rep_{cv_repeat_num}']['y_train_indices']=y_train.index.tolist()
                                    training_results['cv_results'][f'cv_rep_{cv_repeat_num}']['y_test_indices']=y_test.index.tolist()
                                    training_results['cv_results'][f'cv_rep_{cv_repeat_num}']['rand_state']=rand_state

                                    ## LOOP OVER THE PARAMETERS WITH THE TOP N ROC-AUC VALUES DURING PARAMETER SEARCH AND TRAIN MODELS ON WHOLE TRAINING DATA
                                    #top_model_params_in_cv = best_model_params_across_cvs[f'cv_rep_{cv_repeat_num}']

                                    training_results['cv_results'][f'cv_rep_{cv_repeat_num}']['model']={}
                                    
                                    #print(best_model_params)
                                    train_param_comb=f'{data_param_key}_{llm_model_name_with_tag}_{model_name}_{period_end_day}_days_{training_data_type}_{X.shape[1]}_{data_inclusion_type}_autoenc_{autoencoder_merged}'

                                    print(f"Num of CV-repeat:{cv_repeat_num+1} - Model:{model_name} - Data inclusion type:{data_inclusion_type} - Period:{period_end_day} ")

                                    ## If LR, set l1_ratio to 1 to run L1-penalisation , if XGBoost, extract best performing parameter configuration
                                    for n in tqdm(range(num_of_top_models_per_cv),unit="model"):
                                        if model_name=='LogisticRegression':
                                            best_model_params={'l1_ratio':1.0}
                                        
                                        if model_name!='LogisticRegression':
                                            best_model_params=best_model_params_across_cvs[f'cv_rep_{cv_repeat_num}'][n]
                                        
                                        model,cv_roc_auc_scores,label_weights_dict = calc_roc_auc_score_of_model(model_name,
                                                                                                                 X_train,
                                                                                                                 y_train,
                                                                                                                 train_params['k_folds'],
                                                                                                                 train_params['random_state'],
                                                                                                                 outcome_label,
                                                                                                                 best_model_params,
                                                                                                                 train_params['weight_by_label_freq'],
                                                                                                                 train_params,
                                                                                                                 dense_network_params,
                                                                                                                 train_param_comb)

                                        ## Save the best model's ROC-AUC scores as CV validation ROC-AUC
                                        #. + Savel 'label_weights_dict' ==> it depends on the train-test split, so label_weights_dict is the same for all
                                        #. num_of_top_models_per_cv models
                                        if n==0:
                                            training_results['cv_results'][f'cv_rep_{cv_repeat_num}']['cv_roc_auc_scores']=cv_roc_auc_scores 
                                            training_results['cv_results'][f'cv_rep_{cv_repeat_num}']['label_weights_dict']=label_weights_dict
                                        
                                        training_results['cv_results'][f'cv_rep_{cv_repeat_num}']['model'][n]=model

                              
        
                                ## Save training results
                                fn=f'../data/{data_param_key}_{llm_model_name_with_tag}_{model_name}_{period_end_day}_days_{training_data_type}_{X.shape[1]}_{data_inclusion_type}_autoenc_{autoencoder_merged}_vars_training_results.pickle'                                
                                with open(fn, 'wb') as handle:
                                    pickle.dump(training_results, handle)
        
loop_time=time.time()
print('Training duration:')
print_elapsed_time(start,loop_time)  
                                    



