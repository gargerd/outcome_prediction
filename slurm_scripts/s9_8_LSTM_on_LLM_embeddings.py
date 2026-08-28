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

from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from datasets import Dataset 
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, average_precision_score

import io
import pickle
import torch

# Monkey-patch: force all tensor loads to map to CPU
def _cpu_load_from_bytes(b):
    # this is the same as torch.load(...), but we inject map_location='cpu'
    return torch.load(io.BytesIO(b), map_location='cpu')




from s9_8_LSTM_on_LLM_embeddings_functions import *



### EXTRACT PARAMETERS FOR PARAMETER SEARCH FROM ARGPARSE
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", help="One of the dataset names from the keys of parameters_for_analysis dictionar (see below)")
parser.add_argument("--data_inclusion_type", help="One of the 3 data inclusion types: ['baseline_last_day','baseline_vars','all_days']")
#parser.add_argument("--model", help="One of the 2 models: ['XGBoost','LogisticRegression']")
parser.add_argument("--period_end_day", help="One of the 6 periods: [baseline,31,62,93,125,160,'all']")
#parser.add_argument("--pool_method",help="Pooling methods: ['mean_pooling','attention_pooling']")


args = parser.parse_args()
dataset_name_=[args.dataset_name]
data_inclusion_type_ = [args.data_inclusion_type]
#model_names_ = [args.model]
period_end_day_=[args.period_end_day if args.period_end_day in ['baseline','all'] else int(args.period_end_day)]
#pool_methods_=[args.pool_method]






parameters_for_analysis={'tb21_22_2984_pats_22_vars_result_at_end_of_treatment':{
                            'fn':'tb21_22_2984_pats_22_vars_result_at_end_of_treatment',
                            'result_cat':'RESULT_AT_END_OF_TREATMENT'},
            
                        'tb21_22_2984_pats_22_vars_relapse':{
                            'fn':'tb21_22_2984_pats_22_vars_result_at_end_of_treatment',
                            'result_cat':'RELAPSE'}, 


                         
                        'tb21_22_2984_pats_22_vars_result_at_end_of_treatment_dr_reg_per_arm':{
                            'fn':'tb21_22_2984_pats_22_vars_result_at_end_of_treatment',
                             'pat_ids_fn':'tb21_22_2984_pats_22_vars_result_at_end_of_treatment',
                            'result_cat':'RESULT_AT_END_OF_TREATMENT'},
            
                        'tb21_22_2984_pats_22_vars_relapse_dr_reg_per_arm':{
                            'fn':'tb21_22_2984_pats_22_vars_result_at_end_of_treatment',
                            'pat_ids_fn':'tb21_22_2984_pats_22_vars_relapse',
                            'result_cat':'RELAPSE'}, 

                        'tb21_22_2984_pats_22_vars_relapse_without_dr_reg':{
                            'fn':'tb21_22_2984_pats_22_vars_relapse_without_dr_reg',#_wo_dr_reg',
                            'X_fn':'tb21_22_2984_pats_22_vars_result_at_end_of_treatment',
                            'pat_ids_fn':'tb21_22_2984_pats_22_vars_relapse',
                            'result_cat':'RELAPSE'},

                          'tb21_22_2984_pats_22_vars_result_at_end_of_treatment_with_arm':{
                            'fn':'tb21_22_2984_pats_22_vars_result_at_end_of_treatment',
                               'X_fn':'tb21_22_2984_pats_22_vars_result_at_end_of_treatment',
                            'pat_ids_fn':'tb21_22_2984_pats_22_vars_relapse',
                            'result_cat':'RESULT_AT_END_OF_TREATMENT'},
            
                        'tb21_22_2984_pats_22_vars_relapse_with_arm':{
                            'fn':'tb21_22_2984_pats_22_vars_result_at_end_of_treatment',
                             'X_fn':'tb21_22_2984_pats_22_vars_result_at_end_of_treatment',
                            'pat_ids_fn':'tb21_22_2984_pats_22_vars_relapse',
                            'result_cat':'RELAPSE'}, 


                           'tb21_22_2984_pats_22_vars_raw_pred_prob_norm_loss':{
                            'fn':'tb21_22_2984_pats_22_vars_result_at_end_of_treatment',
                            'result_cat':'raw_pred_prob_norm'},
                         
                        'tb21_22_2984_pats_22_vars_llm_pred_prob_norm_loss':{
                            'fn':'tb21_22_2984_pats_22_vars_result_at_end_of_treatment',
                            'result_cat':'llm_pred_prob_norm'},

                 

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



llm_model=load_llm()



### DEFINE training parameters


from sklearn.model_selection import train_test_split
import warnings
from tqdm import tqdm

autoenc_merged_bool_list = [False, True]
outcome_df=pd.read_csv('../../data/tb_1018_20_21_22_30_outcome.csv.gz',index_col=0)
outcome_df=outcome_df.set_index('USUBJID',drop=True)

llm_model_names=['BioMistral/BioMistral-7B',"epfl-llm/meditron-70b","epfl-llm/meditron-7b",
                 "google/long-t5-local-base",'google/long-t5-tglobal-large',
                'text-embedding-3-small']

fine_tuned_tags=['base','fine_tuned']
model_names=['XGBoost','GradientBoost','LogisticRegression','RandomForest']#,'SVC','KNN']
model_names=['XGBoost','LogisticRegression','Dense'][:]#,'SVC','KNN']

#model_names=['LogisticRegression','XGBoost'][:]
training_data_types=['full','pca']
#data_inclusion_types=['baseline_vars','baseline_vars_ext','all_days'][:]
data_inclusion_types=['baseline_last_day','baseline_vars','all_days'][:]

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
              'label2id':label2id,
             'num_epochs':15,
            'hidden_dim':256,
            'num_layers':1,
            'batch_size':64,
            'chunk_len':4096,#1024,
            'use_focal_loss':False,
            'focal_gamma':2.0,
            'focal_alpha':0.25,
            'llm_batch_size':1,
            'patience':3,
              'base_lr':1e-4,
              'label_weight_inv_freq':False,
            #'input_dim':llm_model.config.hidden_size,                
              'scheduler_name':'plateau', #'cosine'
            'device' : "cuda"}


## Extract pats with relapse_df
pats_with_relapse_df=extract_21_22_relapse_pats()

## Subset outcome dataframe to patient considered
eot_outcome_df=outcome_df.loc[:,'RESULT_AT_END_OF_TREATMENT']
eot_outcome_df=eot_outcome_df.replace(label2id)



#### RUN TRAINING

import warnings
warnings.filterwarnings("ignore")
import copy

import time    
start=time.time()

ds_types_merged=True

for data_param_key in dataset_name_:

    ## LOAD FINAL PATIENT IDS FOR ANALYSIS, SAVED DURING PREPROCESSING OF THE BASELINE MODELS IN NOTEBOOK S9_3
    if 'pat_ids_fn' in parameters_for_analysis[data_param_key].keys():
        fn=f"../../data/{parameters_for_analysis[data_param_key]['pat_ids_fn']}_final_pat_ids_for_analysis.pickle"
    else:  
        fn=f'../../data/{data_param_key}_final_pat_ids_for_analysis.pickle'
    with open(fn, 'rb') as handle:
        final_pat_ids_for_analysis=pickle.load(handle)
    
    print(data_param_key)
    #fn='../../data/'+data_param_key+'_all_data_concat.csv.gz'
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

                    print('data loaded!')
                    if 'X_fn' in parameters_for_analysis[data_param_key].keys():
                        fn='../../data/'+parameters_for_analysis[data_param_key]['X_fn']+'_all_data_concat.csv.gz'
                    else:
                        fn='../../data/'+parameters_for_analysis[data_param_key]['fn']+'_all_data_concat.csv.gz'
                        
                    X=pd.read_csv(fn,index_col=0,low_memory=False)

                    ## Return dataframe with the outcome label
                    pat_ids,y,target_df,outcome_label = return_predict_label_dataframe(parameters_for_analysis,data_param_key,X,
                                                                                  outcome_df,outcome_label,model_names)
                                             
            

                    ## Subset initial therapy last day dataframe to all patient considered in analysis
                    #init_ther_df=last_initial_therapy_day_df.loc[pat_ids,:]
                    last_init_ther_days = extract_last_init_therapy_day_from_drug_regimen(pat_ids)

    
                    df_=y.reset_index()#
                    df_['STUDYID']=df_['USUBJID'].str.split('/',expand=True)[0].values
                    #print(pd.crosstab(df_.loc[df_['USUBJID'].isin(pat_ids_),'STUDYID'],df_.loc[df_['USUBJID'].isin(pat_ids_),outcome_label]))
                    
                    for training_data_type in training_data_types[:1]:
    
                        ## TOKENIZE DATASET
                        prompt=create_prompt(data_inclusion_type)

                        pat_ids,all_ds_types,all_texts,all_labels = load_input_texts_and_labels(ds_types_merged,
                                                                                                data_param_key,
                                                                                                target_df,
                                                                                                prompt,
                                                                                                period_end_day,
                                                                                                data_inclusion_type,
                                                                                               outcome_label)

                        
                        #all_dataset_raw=Dataset.from_dict({"text": all_texts[:], 
                                                          # "labels": all_labels[:1],
                        #                                   'pat_ids':[*pat_ids][:]}).with_format("torch")
                        #all_dataset_tokenized=all_dataset_raw.map(lambda x: tokenizer(x['text'],truncation=False, padding=False, return_tensors='pt'),batched=False)
                        #all_loader=DataLoader(all_dataset_tokenized, batch_size=1)
                    
                                  
                        print(f'+++++++\nRunning training with {training_data_type} model: {X.shape[1]} vars\n+++++++')

                        
                        #for model_name in model_names[:]:
                        #for n, model_name in enumerate(tqdm(model_names[2:], desc="Processing", unit="model")):
                        print('\n=====================')
                        #print(model_name)

                        training_results={}
                        training_results['train_params']=train_params
                        training_results['cv_results']={}

               
                            
                        for cv_repeat_num in tqdm(range(train_params['num_cv_repeats']),desc="Processing", unit="cv_repeat"):
                        #for cv_repeat_num in tqdm(range(1),desc="Processing", unit="cv_repeat"):
                        
                            rand_state=train_params['random_state'] + cv_repeat_num

                            X_train_ids=final_pat_ids_for_analysis[period_end_day][cv_repeat_num]['X_train_ids']
                            X_test_ids=final_pat_ids_for_analysis[period_end_day][cv_repeat_num]['X_test_ids']
                            
                            #X_train=X.loc[list(set(X.index)&set(X_train_ids)),:]
                            #X_test=X.loc[list(set(X.index)&set(X_test_ids)),:]

                            y_train=y.loc[(X_train_ids),:]
                            y_test=y.loc[(X_test_ids),:]
                            
                            #if model_name=='LogisticRegression':
                           # X_train, X_test = scale_by_training_data(X_train, X_test)

                            print('train ids',len(X_train_ids))
                            print('test ids',len(X_test_ids))



                            y_for_strat=y_train[outcome_label].astype(str) + "_" + y_train.index.get_level_values('STUDYID')

                                                        
                            X_train_inner_ids, X_val_inner_ids, \
                                y_train_inner_labels, y_val_inner_labels = train_test_split(X_train_ids,
                                                                                        y_train,
                                                                                        test_size=0.2,
                                                                                        stratify=y_for_strat,
                                                                                        random_state=rand_state)
                                                                                    
                        
                            
                            model_id=f"{llm_model_name}_{fine_tuned_tag}"
                            
                            
                            
                            ## SUBSET TOKENIZED DATA TO THE PATIENTS IN THE TESTING COHORT + LOAD THEM AS A DATALOADER OBJECT
                            #train_inner_data_tokenized = all_dataset_tokenized.filter(lambda x: x["pat_ids"] in  X_train_inner_ids)
                            #val_inner_data_tokenized = all_dataset_tokenized.filter(lambda x: x["pat_ids"] in  X_val_inner_ids)

                            #[for text,lab_,id_ in zip(all_texts[:],all_labels[:],[*pat_ids][:])

                            train_data=np.array([(text,int(lab_),id_) for text,lab_,id_ in zip(all_texts[:],all_labels[:],[*pat_ids][:]) if id_ in X_train_inner_ids])
                            train_texts = train_data[:,0].tolist()
                            train_labels =train_data[:,1].astype(int).tolist()
                            train_ids_ = train_data[:,2].tolist()

                            val_data=np.array([(text,int(lab_),id_) for text,lab_,id_ in zip(all_texts[:],all_labels[:],[*pat_ids][:]) if id_ in X_val_inner_ids])
                            val_texts = val_data[:,0].tolist()
                            val_labels = val_data[:,1].astype(int).tolist()
                            val_ids_ = val_data[:,2].tolist()

                    
                                                       
                            num_epochs=train_params['num_epochs']
                            hidden_dim=train_params['hidden_dim']
                            num_layers=train_params['num_layers']
                            batch_size=train_params['batch_size']
                            chunk_len=train_params['chunk_len']
                            use_focal_loss=train_params['use_focal_loss']
                            focal_gamma=train_params['focal_gamma']
                            focal_alpha=train_params['focal_alpha']
                            llm_batch_size=train_params['llm_batch_size']
                            patience=train_params['patience']
                            input_dim=llm_model.config.hidden_size          
                            device=train_params['device']
                            base_lr=train_params['base_lr']
                            label_weight_inv_freq=train_params['label_weight_inv_freq']
                            scheduler_name = train_params['scheduler_name']
                                                         
                            model, history = train_full_pipeline(
                                                train_texts=train_texts,
                                                train_labels=train_labels,
                                                val_texts=val_texts,
                                                val_labels=val_labels,
                                                num_epochs=num_epochs,
                                                chunk_len=chunk_len,
                                                batch_size=batch_size,
                                                hidden_dim=hidden_dim,
                                                #pos_weight=pos_weight,
                                                use_focal_loss=use_focal_loss,
                                                focal_gamma=focal_gamma,
                                                focal_alpha=focal_alpha,
                                                llm_batch_size=llm_batch_size,
                                                patience=patience,
                                                base_lr=base_lr,
                                                llm_model=llm_model,
                                                device=device,
                                                scheduler_name = scheduler_name,
                                                num_layers=num_layers,
                                                label_weight_inv_freq=label_weight_inv_freq,
                                                input_dim=input_dim)
                            # after training
                            peak_bytes = torch.cuda.max_memory_allocated(device=0)
                            print("Peak PyTorch memory allocated (GB):", peak_bytes / 1024**3)
                            #print(history)

                            
                            test_data=np.array([(text,int(lab_),id_) for text,lab_,id_ in zip(all_texts[:],all_labels[:],[*pat_ids][:]) if id_ in X_test_ids])
                            test_texts = test_data[:,0].tolist()
                            test_labels =test_data[:,1].astype(int).tolist()
                            test_ids_ = test_data[:,2].tolist()


                            pred_probs, true_labels = run_model_on_test_set(
                                                                        model=model,
                                                                        llm_model=llm_model,
                                                                        tokenizer=tokenizer,
                                                                        test_texts=test_texts,
                                                                        test_labels=test_labels,
                                                                        device=device,
                                                                        chunk_len=chunk_len,
                                                                        llm_batch_size=llm_batch_size,
                                                                        batch_size=batch_size)
                                                                                                                       
                                                                                                                        
                                                  
                            ## SAVE PARAMETERRS OF CV-SPLIT
                            print(f"Num of CV-repeat:{cv_repeat_num+1}")
                            training_results['cv_results'][f'cv_rep_{cv_repeat_num}']={}

                            training_results['cv_results'][f'cv_rep_{cv_repeat_num}']['X_train_inner_ids']=train_ids_
                            training_results['cv_results'][f'cv_rep_{cv_repeat_num}']['X_val_inner_ids']=val_ids_
                            training_results['cv_results'][f'cv_rep_{cv_repeat_num}']['X_test_ids']=test_ids_
  
                            #training_results['cv_results'][f'cv_rep_{cv_repeat_num}']['y_train_indices']=y_train.index.tolist()
                            #training_results['cv_results'][f'cv_rep_{cv_repeat_num}']['y_test_indices']=y_test.index.tolist()
                            training_results['cv_results'][f'cv_rep_{cv_repeat_num}']['rand_state']=rand_state

                  
                            #training_results['cv_results'][f'cv_rep_{cv_repeat_num}']['cv_roc_auc_scores']=cv_roc_auc_scores 
                            #training_results['cv_results'][f'cv_rep_{cv_repeat_num}']['label_weights_dict']=label_weights_dict


                            model_saving_dict = {"bilstm_state_dict": model.state_dict(),
                                                "bilstm_config": {
                                                    "input_dim": input_dim,
                                                    "hidden_dim": hidden_dim,
                                                    "num_layers": num_layers,
                                                    "bidirectional": True,
                                                    "dropout": 0.1,
                                                    "attention_dim": hidden_dim,
                                                },
                                                "history": history,
                                                }

                            training_results['cv_results'][f'cv_rep_{cv_repeat_num}']['model']=model_saving_dict


                            
                            training_results['cv_results'][f'cv_rep_{cv_repeat_num}']['test_pred_probs']=pred_probs
                            training_results['cv_results'][f'cv_rep_{cv_repeat_num}']['test_true_labels']=true_labels

                          
                          
                      

                        ## Save training results
                        save_dir=f'../../data/LSTM_on_LLM_embeddings'
                        os.makedirs(save_dir,exist_ok=True)
                        
                        fn=os.path.join(save_dir,
                                        f"{data_param_key}_{llm_model_name_with_tag}_{period_end_day}_days_{training_data_type}_{data_inclusion_type}_vars_training_results.pickle")
                        with open(fn, 'wb') as handle:
                            pickle.dump(training_results, handle)
    
loop_time=time.time()
print('Training duration:')
print_elapsed_time(start,loop_time)  
                                

                  