import pandas as pd
import pickle
import json
from matplotlib import pyplot as plt
import torch
import numpy as np
import os
import itertools
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm
import copy
import re

from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from datasets import Dataset 
from torch.utils.data import DataLoader, TensorDataset

from s9_6_LLM_integrated_gradients_LR_functions import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


### EXTRACT PARAMETERS FOR PARAMETER SEARCH FROM ARGPARSE
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", help="One of the dataset names from the keys of parameters_for_analysis dictionar (see below)")
parser.add_argument("--data_inclusion_type", help="One of the 3 data inclusion types: ['baseline_last_day','baseline_vars','all_days']")
parser.add_argument("--model", help="One of the 2 models: ['XGBoost','LogisticRegression']")
parser.add_argument("--period_end_day", 
                    nargs='+',  # ← this makes it accept one or more values
                    help="One of the 6 periods: [baseline,31,62,93,125,160,'all']")
parser.add_argument("--internal_batch_size", help="Internal batch size for captum's LayerIntegratedGradient. For baseline_last_day timepoint set it to 4, for timepoint with more input tokens (>~1200) set it to 1 to avoid out-of-memory error")
parser.add_argument("--pool_method",help="Pooling methods: ['mean_pooling','attention_pooling']")


args = parser.parse_args()
dataset_name_=[args.dataset_name]
data_inclusion_type_ = [args.data_inclusion_type]
model_names_ = [args.model]
#period_end_day_=[args.period_end_day if args.period_end_day in ['baseline','all'] else int(args.period_end_day)]
period_end_day_ = [p if p in ['baseline', 'all'] else int(p) for p in args.period_end_day]
internal_batch_size=int(args.internal_batch_size)
pool_methods_=[args.pool_method]


# DEFINE DATASET NAMES + FUNCTIONS FOR LOADING
parameters_for_analysis={'tb21_22_2984_pats_22_vars_result_at_end_of_treatment':{
                            'fn':'tb21_22_2984_pats_22_vars_result_at_end_of_treatment',
                            'result_cat':'RESULT_AT_END_OF_TREATMENT'},
            
                        'tb21_22_2984_pats_22_vars_relapse':{
                            'fn':'tb21_22_2984_pats_22_vars_result_at_end_of_treatment',
                            'result_cat':'RELAPSE'}, 

                        'tb21_22_2984_pats_22_vars_relapse_ext_pats':{
                            'fn':'tb21_22_2984_pats_22_vars_result_at_end_of_treatment',
                            'result_cat':'RELAPSE'}, 

                         'tb21_22_2984_pats_22_vars_result_at_end_of_treatment_without_dr_reg':{
                            'result_cat':'RESULT_AT_END_OF_TREATMENT',
                              'X_fn':'tb21_22_2984_pats_22_vars_result_at_end_of_treatment',
                            'fn':'tb21_22_2984_pats_22_vars_result_at_end_of_treatment_without_dr_reg',
                            'pat_ids_fn':'tb21_22_2984_pats_22_vars_result_at_end_of_treatment',
                         },


                        'tb21_22_2984_pats_22_vars_relapse_without_dr_reg_ext_pats':{
                            'X_fn':'tb21_22_2984_pats_22_vars_result_at_end_of_treatment',
                            'fn':'tb21_22_2984_pats_22_vars_relapse_without_dr_reg_ext_pats',
                            'pat_ids_fn':'tb21_22_2984_pats_22_vars_relapse_ext_pats',
                            'result_cat':'RELAPSE',
                            'validate_on_rifaquin':True},

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


## Load embeddings of LLM models into a dictionary for faster inference later
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
    
    outcome_label = data_param_key.split('vars_')[-1].upper()
    
    for llm_model_name in llm_model_names[:1]:
        
        for fine_tuned_tag in fine_tuned_tags[:1]:
            
            if 'text-embedding' in llm_model_name:
                llm_model_name_with_tag = llm_model_name
            else:
                llm_model_name_with_tag = '_'.join([llm_model_name.split('/')[-1], fine_tuned_tag])
                
            for period_end_day in period_end_day_:
                
                for data_inclusion_type in data_inclusion_type_:
                    
                    for autoencoder_merged in autoenc_merged_bool_list[:1]:
                        
                        for pool_method in pool_methods_:
                            combinations.append((
                                data_param_key, llm_model_name, fine_tuned_tag,
                                llm_model_name_with_tag, period_end_day,
                                data_inclusion_type, autoencoder_merged,
                                pool_method
                            ))


# Iterate with progress bar
for combo in tqdm(combinations, desc="Loading embeddings", unit="set"):
    data_param_key, llm_model_name, fine_tuned_tag, \
                    llm_model_name_with_tag, period_end_day, \
                        data_inclusion_type, autoencoder_merged, pool_method = combo
    
    embed_dict.setdefault(data_param_key, {})
    embed_dict[data_param_key].setdefault(llm_model_name_with_tag, {})
    embed_dict[data_param_key][llm_model_name_with_tag].setdefault(period_end_day, {})
    embed_dict[data_param_key][llm_model_name_with_tag][period_end_day].setdefault(data_inclusion_type, {})
    embed_dict[data_param_key][llm_model_name_with_tag][period_end_day][data_inclusion_type].setdefault(autoencoder_merged, {})

    print(f"\n→ Loading: {data_param_key} | {llm_model_name_with_tag} | {period_end_day} | {data_inclusion_type} | autoenc={autoencoder_merged} | {pool_method} ")

    try:
        start = time.time()
        df_pt = load_input_embeddings_of_model(
            data_param_key, llm_model_name, fine_tuned_tag,
            period_end_day, llm_model_name_with_tag,data_inclusion_type,pool_method,
            autoencoder_merged)
        
        elapsed = time.time() - start
        print(f"Loaded in {elapsed:.2f} seconds.")
    except FileNotFoundError:
        warnings.warn(f"Embeddings missing for: {data_param_key} / {llm_model_name_with_tag}. Skipping.")
        continue

    embed_dict[data_param_key][llm_model_name_with_tag][period_end_day][data_inclusion_type][autoencoder_merged][pool_method] = df_pt
    #print('Inf present:',any(df_pt==np.inf))
    #print('-Inf present:',any(df_pt==-np.inf))
    assert not np.isinf(df_pt.values).any(), "There are still Inf values!"
    assert not np.isnan(df_pt.values).any(), "There are still NaN values!"



#### ============== LOAD LLM =====================

from peft import LoraConfig, TaskType,get_peft_model
'''
LORA taks types:

SEQ_CLS = "SEQ_CLS"
SEQ_2_SEQ_LM = "SEQ_2_SEQ_LM"
CAUSAL_LM = "CAUSAL_LM"
TOKEN_CLS = "TOKEN_CLS"
QUESTION_ANS = "QUESTION_ANS"
FEATURE_EXTRACTION = "FEATURE_EXTRACTION"
'''

task_type_dict={'AutoModelForCausalLM':'CAUSAL_LM','AutoModelForSequenceClassification':'SEQ_CLS'}

#peft_config = LoraConfig(task_type=task_type_dict[model_type], inference_mode=False, r=8, lora_alpha=8, lora_dropout=0.1)

## READ HUGGING FACE TOKEN (needed for gated models e.g. meditron)
from huggingface_hub import login
hf_token_path = '../../api_tokens/hf_token.txt'
hf_token = None
if os.path.exists(hf_token_path):
    with open(hf_token_path, 'r') as f:
        hf_token = f.read().strip()
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token)
        print('Hugging Face token loaded and logged in')
else:
    print(f'No HF token found at {hf_token_path} — gated models (e.g. meditron) will fail without one')


from transformers import AutoModelForSequenceClassification,LongT5EncoderModel, AutoConfig
from peft import LoraConfig, TaskType,get_peft_model,PeftModel

# Define the path where your model is saved
cache_dir='../../huggingface_cache'


## Set up prediction labels
id2label={0: "FAVOURABLE", 1: "UNFAVOURABLE"}
label2id={"FAVOURABLE": 0, "UNFAVOURABLE": 1}

llm_model_names=['BioMistral/BioMistral-7B',"epfl-llm/meditron-7b",'FremyCompany/BioLORD-2023',
                 "google/long-t5-local-base",'google/long-t5-tglobal-large',"epfl-llm/meditron-70b"]
num_of_epochs=4
fine_tuned_tags=['base','fine_tuned']

print('\n Loading LLM\n')

for llm_model_name in llm_model_names[:1]:
    print(llm_model_name)
        
    for fine_tuned_tag in fine_tuned_tags[:1]:

        
        ## LOAD BASE MODEL ==> precision: torch_dtype torch.float16
        
        if 'long-t5' in llm_model_name:
            config = AutoConfig.from_pretrained(llm_model_name, tie_word_embeddings=True)
            tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
            model = LongT5EncoderModel.from_pretrained(llm_model_name,
                                                             cache_dir=cache_dir,
                                                             #output_hidden_states=True,
                                                             #torch_dtype=torch.float16,                                                        
                                                             #num_labels=len(id2label.keys()),
                                                             #id2label=id2label, 
                                                             #label2id=label2id,
                                                             config=config,
                                                             device_map=0
                                                             #device_map={"":0}#'auto',
                                                       ).to(device)
                                                             #pad_token_id=2)
            model.resize_token_embeddings(len(tokenizer))
        
        elif 'meditron-70b' in llm_model_name:
            from transformers import AutoModelForSequenceClassification, BitsAndBytesConfig,AutoModelForCausalLM
            quant_config=BitsAndBytesConfig(
                load_in_4bit=True,
                load_in_8bit=False,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=False
            )
            model=AutoModelForSequenceClassification.from_pretrained(llm_model_name,
                                                                     cache_dir=cache_dir,
                                                                     output_hidden_states=True,
                                                                     #output_attentions=True,
                                                                     #torch_dtype=torch.float16,                                                        
                                                                     num_labels=len(id2label.keys()),
                                                                     id2label=id2label, 
                                                                     label2id=label2id,
                                                                     device_map='auto',
                                                                     quantization_config=quant_config,
                                                                     pad_token_id=2)

        else:
            model=AutoModelForSequenceClassification.from_pretrained(llm_model_name,
                                                                     cache_dir=cache_dir,
                                                                     output_hidden_states=True,
                                                                     #output_attentions=True,
                                                                     torch_dtype=torch.float16,                                                        
                                                                     num_labels=len(id2label.keys()),
                                                                     id2label=id2label, 
                                                                     label2id=label2id,
                                                                     device_map='auto',
                                                                     pad_token_id=2,
                                                                     force_download=False, 
                                                                     resume_download=False)
        
        
        
        print(f'{llm_model_name} base model loaded')

        ## IF INFERENCE WISHED BY FINE-TUNED MODEL, ADD & MERGE FINE-TUNED PEFT MODEL ADAPTER
        peft_model_path=f"../../data/{llm_model_name}_{fine_tuned_tag}_num_of_epochs_{num_of_epochs}_{period_end_day}_days"
        peft_model_path=f"../../data/{llm_model_name}_{fine_tuned_tag}_num_of_epochs_{num_of_epochs}"#_{period_end_day}_days"
        
        if fine_tuned_tag=='fine_tuned':
            model=PeftModel.from_pretrained(model,peft_model_path)
            model=model.merge_and_unload()
            
            print(f'{llm_model_name}_{fine_tuned_tag} PEFT fine-tuned adapter added')



######================== LOAD PREDICTION LABELS ================
outcome_df=pd.read_csv('../../data/tb_1018_20_21_22_30_outcome.csv.gz',index_col=0)
outcome_df=outcome_df.set_index('USUBJID',drop=True)


## LOOP THROUGH THE DIFFERENT DATASETS AND ALL THE DIFFERENT DATA LAYERS, AND CONVERT TABULAR DATA INTO DICTIONARY OF STRINGS
for key in [*parameters_for_analysis][:1]:
    ## Load data
    fn='../../data/'+key+'_all_data_concat.csv.gz'
    data_conc=pd.read_csv(fn,low_memory=False,index_col=0,usecols=range(5))
    pat_ids=data_conc['USUBJID'].unique().tolist()
    #del data_conc
    

    ## Extract outcome label and convert FAVOURABLE/UNFAVOURABLE to 0/1
    outcome_label=key.split('vars_')[-1].upper()
    #outcome_df[outcome_label]=outcome_df[outcome_label].replace({'FAVOURABLE':1,'UNFAVOURABLE':0},regex=False).values
    target_df=outcome_df.loc[pat_ids,outcome_label]
    #print(target_df.value_counts(dropna=False))
    #print(key)



####============= INIT TOKENIZER =================

## LOAD TOKENIZER AND TOKENIZE INPUT TEXTS
#tokenizer=load_huggingface_tokenizer(model_name)
tokenizer = AutoTokenizer.from_pretrained(llm_model_name,
                                          #force_download=False,
                                            #truncation_side="left",
                                            #padding_side="right",
                                            add_eos_token=True,
                                            add_bos_token=True
                                         )
        
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id




####============= RUN IG =================

period_end_days_for_plot=period_end_days[:]

llm_model_names=['BioMistral/BioMistral-7B',"epfl-llm/meditron-7b"]
fine_tuned_tags=['base','fine_tuned']
#model_names=['XGBoost','GradientBoost','LogisticRegression','RandomForest','SVC','KNN']
training_data_types=['full','pca']

llm_model_names=['BioMistral/BioMistral-7B',"epfl-llm/meditron-70b","epfl-llm/meditron-7b",
                 "google/long-t5-local-base",'google/long-t5-tglobal-large',
                'text-embedding-3-small']

suptitle_dict={'tb21_22_18_3079_pats_22_vars_result_at_end_of_treatment':'REMox + OFLOTUB',
              'tb21_1405_pats_41_vars_result_at_end_of_treatment':'REMox'}

model_names=['XGBoost','LogisticRegression',]
## SET UP DEVICE AND LOAD PRETRAINED MODEL FROM HUGGINGFACE
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

plot_split_by_studies=True
ds_types_merged=True


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


#chunk_token_length = 2048
#chunk_token_length = 2560
chunk_token_length = 2560
chunk_token_length = 10000
chunk_overlap = 64
n_steps = 24
#internal_batch_size = 1



for data_param_key in dataset_name_:

    ## LOAD FINAL PATIENT IDS FOR ANALYSIS, SAVED DURING PREPROCESSING OF THE BASELINE MODELS IN NOTEBOOK S9_3
    final_pat_ids_for_analysis = load_final_patient_for_analysis(parameters_for_analysis=parameters_for_analysis,
                                                                 data_param_key=data_param_key)
    

    #print(data_param_key)
    if 'X_fn' in parameters_for_analysis[data_param_key].keys():
        fn='../../data/'+parameters_for_analysis[data_param_key]['X_fn']+'_all_data_concat.csv.gz'
    else:
        fn='../../data/'+parameters_for_analysis[data_param_key]['fn']+'_all_data_concat.csv.gz'
    X_=pd.read_csv(fn,index_col=0,low_memory=False)
    
    print(data_param_key)
    
    outcome_label = parameters_for_analysis[data_param_key]['result_cat']
        
    for llm_model_name in llm_model_names[:1]:
        
        for fine_tuned_tag in fine_tuned_tags[:1]:
            
            
            if 'text-embedding' in llm_model_name:
                llm_model_name_with_tag=llm_model_name
            else:
                llm_model_name_with_tag='_'.join([llm_model_name.split('/')[-1],fine_tuned_tag])
            
            

            for data_inclusion_type in data_inclusion_type_:

                for training_data_type in training_data_types[:1]:
                

                    for autoencoder_merged in autoenc_merged_bool_list[:1]:
                    
                        for period_end_day in period_end_day_:
    
                            if outcome_label=='RESULT_AT_END_OF_TREATMENT' and period_end_day=='all':
                                print('Skipping all period!')
                                continue
                            

                            for pool_method in pool_methods_:
    
                                ## Load embeddings created by LLM model
                                print(f'Loading embeddings of {data_param_key} data / {llm_model_name_with_tag} /{period_end_day} /{data_inclusion_type} /autoenc {autoencoder_merged} / {pool_method} model')
            
                                ## Load embeddings created by LLM model
                                #print(f'Loading embeddings of {llm_model_name_with_tag} model')
                                try:
                                    df=embed_dict[data_param_key][llm_model_name_with_tag][period_end_day][data_inclusion_type][autoencoder_merged][pool_method]
                                except KeyError:
                                    warnings.warn(f"Embeddings of {parameters_for_analysis[data_param_key]['fn']} data / {llm_model_name_with_tag} model/ {period_end_day} days/ {data_inclusion_type}/ autoenc {autoencoder_merged} /{pool_method} not found! Check if they have been created. Skipping to next item.")
                                    continue

                            
                                period_num=period_end_days.index(period_end_day)
    
                    
    
    
    
    
                                ## LOAD PATIENT SENTENCES
                                prompt=create_prompt()        
                                pat_ids,all_ds_types,all_texts,all_labels = load_input_texts_and_labels(ds_types_merged,
                                                                                                        data_param_key,
                                                                                                        target_df,
                                                                                                        prompt,
                                                                                                        period_end_day,
                                                                                                        data_inclusion_type)
        
        
                                
                                ## TOKENIZE DATASET
                                all_dataset_raw=Dataset.from_dict({"text": all_texts[:], "labels": all_labels[:],'pat_ids':[*pat_ids][:]})#.with_format("torch")
                                all_dataset_tokenized=all_dataset_raw.map(lambda x: tokenizer(x['text'],truncation=False, padding=False, return_tensors='pt'),batched=False)
                                all_dataset_tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels',"text",'pat_ids'])
                                #all_loader=DataLoader(all_dataset_tokenized, batch_size=1)
                                                
                            
                                
        
                                for model_name in model_names_:
                                    #print('model_name',model_name)
            
                                    #fn=f'../../data/{data_param_key}_{llm_model_name_with_tag}_{model_name}_{period_end_day}_days_{training_data_type}_{X.shape[1]}_vars_training_results.pickle'
                                    #fn=f'../../data/{data_param_key}_{llm_model_name_with_tag}_{model_name}_{period_end_day}_days_{training_data_type}_{X.shape[1]}_{data_inclusion_type}vars_training_results.pickle'
                                    fn=f'../../data/{data_param_key}_{llm_model_name_with_tag}_{model_name}_{period_end_day}_days_{training_data_type}_4096_{data_inclusion_type}_autoenc_{autoencoder_merged}_{pool_method}_vars_training_results.pickle'                                
                                    with open(fn, 'rb') as handle:
                                        training_results=pickle.load(handle)
                                        print('loaded training results of', fn)
            
                                    cv_results=training_results['cv_results']
        
                                    X=df.copy()
        
                                    ## LOAD TRAIN-TEST DATA AND SCALE BY TRAINING DATA'S DISTRIBUTION
                                    #for cv_repeat_num in range(len([*cv_results])): 
                                    for cv_repeat_num in tqdm(range(len([*cv_results])), desc="CV_repeat",total=len([*cv_results])):
                                    #for cv_repeat_num in range(1): 
                                    
        
                                        X_train_ids=final_pat_ids_for_analysis[period_end_day][cv_repeat_num]['X_train_ids']
                                        X_test_ids=final_pat_ids_for_analysis[period_end_day][cv_repeat_num]['X_test_ids']
        
                                        X_train_ids_=list(set(X.index)&set(X_train_ids))
                                        X_test_ids_=list(set(X.index)&set(X_test_ids))
                                        
                                        X_train=X.loc[X_train_ids_,:]
                                        X_test=X.loc[X_test_ids_,:]
        
                                        std_scaler = StandardScaler()
                                        std_scaler.fit(X_train)
                                        mean = torch.tensor(std_scaler.mean_, dtype=torch.float32).to(device)  # shape (4096,)
                                        scale = torch.tensor(std_scaler.scale_, dtype=torch.float32).to(device)  # shape (4096,)
    
                                        print('mean.shape',mean.shape,'scale.shape',scale.shape)
                                         
        
                                        #if data_inclusion_type!='all_days' or model_name=='LogisticRegression':
                                        _, X_test = scale_by_training_data(X_train, X_test)
                              
        
                                        ## EXtract the output probabilities for each top n models and average them to get a final prediction probability
                                        if isinstance(cv_results[f'cv_rep_{cv_repeat_num}']['model'],dict):
                                            
                                            for n in range(len(cv_results[f'cv_rep_{cv_repeat_num}']['model'].keys())):
                                                #print(n,'model',model_name)                                        
                                                lr_model=cv_results[f'cv_rep_{cv_repeat_num}']['model'][n]
        
                                                lr_torch = SklearnLRWrapper(lr_model).to(device)
        
                                                
                                                #print(f'Extract embedding of {model_id}/{data_inclusion_type}/')
                                                #with torch.no_grad():
                                                #print(model_id,key,period_end_day)
    
                                                ## SUBSET TOKENIZED DATA TO THE PATIENTS IN THE TESTING COHORT + LOAD THEM AS A DATALOADER OBJECT
                                                test_data_tokenized = all_dataset_tokenized.filter(lambda x: x["pat_ids"] in  X_test_ids_)
                                                
                                                test_loader=DataLoader(test_data_tokenized, batch_size=1)
                                    
                                                for batch in tqdm(test_loader, desc=f"{data_inclusion_type}/{period_end_day}",total=len(test_loader)):
    
                                                    if batch["pat_ids"][0] not in set(X_test_ids_):
                                                        continue
                                                    
                                                    input_text = batch["text"][0]
                                                    pat_id = batch["pat_ids"][0]
                                                    input_ids = batch['input_ids'][0].squeeze(0)
        
                                                    
                                        
                                        
                                                    #try:
                                                    # Compute IG for all embedding dims
                                                    #full_text = f"{instruction} {input_text}"
                                                    #tokenized = tokenizer(input_text, return_tensors="pt", add_special_tokens=True)
                                                    #input_ids = tokenized["input_ids"].squeeze(0)
                                    
                                                    chunks, starts = chunk_token_ids_with_indices(input_ids, max_len=chunk_token_length, 
                                                                                                  overlap=chunk_overlap)
                                                    attr_map = {}
                                                    token_map = {}
                                    
                                                    for chunk_ids, start in zip(chunks, starts):
                                                        #attr_tensor, tokens = run_ig_on_chunk(chunk_ids, model)
                                                        attr_tensor, tokens = run_lig_on_chunk(chunk_ids, model,tokenizer,
                                                                                               n_steps,internal_batch_size,
                                                                                                mean,scale,lr_torch,input_text)
                                                        
                                    
                                                        for i, (score, token) in enumerate(zip(attr_tensor, tokens)):
                                                            global_idx = start + i
                                                            if global_idx not in attr_map:
                                                                attr_map[global_idx] = []
                                                                token_map[global_idx] = token
                                                            attr_map[global_idx].append(score)
                                                        
                                    
                                                    # Aggregate (average) overlapping attributions
                                                    sorted_indices = sorted(attr_map.keys())
                                                    final_attr = torch.stack([torch.stack(attr_map[idx]).mean(dim=0) for idx in sorted_indices])
                                                    final_tokens = [token_map[idx] for idx in sorted_indices]
        
    
                                                    
                                                    ## SAVE ATTRIBUTES AND TOKENS                        
                                                    model_id=f"{llm_model_name}_{fine_tuned_tag}"
                                                    ig_output_dir=f"../../data/integrated_gradients/{data_param_key}_{model_id}_{period_end_day}_days_{data_inclusion_type}_{pool_method}"
                                                    os.makedirs(ig_output_dir,exist_ok=True)
                                                    
                                                    pat_id_=pat_id.replace('/','_')
                                                    attr_fn=os.path.join(ig_output_dir,f"{pat_id_}_CV_rep_{cv_repeat_num}_IG_attr.npz")
                                                    token_attr_sum = final_attr.cpu().numpy().sum(axis=1)  
                                                    np.savez_compressed(attr_fn,attributions=token_attr_sum)
                                                    #np.savez_compressed(attr_fn,attributions=final_attr.cpu().numpy())
                                                    
                                                    ## As tokens are the same across CV-reps, save only once
                                                    #if cv_repeat_num==0:
                                                    token_fn=os.path.join(ig_output_dir,f"{pat_id_}_tokens.npz")        
                                                    np.savez_compressed(token_fn,tokens=np.array(final_tokens))
                                                    
                                                    #print(f"Saved: {pat_id} | Tokens: {len(final_tokens)} | Attr shape: {final_attr.shape}")


                               
 





    