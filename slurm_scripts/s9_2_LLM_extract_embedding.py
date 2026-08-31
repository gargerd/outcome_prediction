
import pandas as pd
import pickle
import json
from matplotlib import pyplot as plt
import torch
import numpy as np
import os
import itertools
import seaborn as sns

from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from datasets import Dataset 
from torch.utils.data import DataLoader, TensorDataset




### EXTRACT PARAMETERS FOR PARAMETER SEARCH FROM ARGPARSE
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", 
                    nargs='+',
                    help="List of the dataset names from the keys of parameters_for_analysis dictionar (see below)")
parser.add_argument("--data_inclusion_type", 
                    nargs='+',  # ← this makes it accept one or more values
                    help="List of data inclusion types: ['baseline_last_day','baseline_vars','all_days']")

parser.add_argument("--pool_methods",
                    nargs='+',  # ← this makes it accept one or more values
                    help="Pooling methods: ['mean_pooling','attention_pooling']")
parser.add_argument("--period_end_day", 
                    nargs='+',  # ← this makes it accept one or more values
                    help="One of the 6 periods: [baseline,31,62,93,125,160,'all']")



args = parser.parse_args()
dataset_name_=[t for t in args.dataset_name]
data_inclusion_types_ = [t for t in args.data_inclusion_type]
pool_methods_ = [t for t in args.pool_methods]
#model_names_ = [args.model]
#period_end_day_=[args.period_end_day if args.period_end_day in ['baseline','all'] else int(args.period_end_day)]
period_end_days_ = [p if p in ['baseline', 'all'] else int(p) for p in args.period_end_day]
#internal_batch_size=int(args.internal_batch_size)





##### ==============  Define dataset names + functions

parameters_for_analysis={'tb21_22_2984_pats_22_vars_result_at_end_of_treatment':{
                            'result_cat':'RESULT_AT_END_OF_TREATMENT',
                            'fn':'tb21_22_2984_pats_22_vars_result_at_end_of_treatment',
                            'selection_method':'patient_clustering',
                            'clust_comb':'2-3-4-5',
                            'graph_metric':None,
                            'num_of_common_vars':22,
                            'training_days':120},
                         
                         'tb21_22_2984_pats_22_vars_relapse_without_dr_reg':{
                            'result_cat':'RESULT_AT_END_OF_TREATMENT',
                            'fn':'tb21_22_2984_pats_22_vars_result_at_end_of_treatment'},

                         'tb21_22_2984_pats_22_vars_result_at_end_of_treatment_without_dr_reg':{
                            'result_cat':'RESULT_AT_END_OF_TREATMENT',
                            'fn':'tb21_22_2984_pats_22_vars_result_at_end_of_treatment'},

                             'tb21_22_2984_pats_22_vars_relapse_without_dr_reg_ext_pats':{
                             'fn':'tb21_22_2984_pats_22_vars_result_at_end_of_treatment',
                            'pat_ids_fn':'tb21_22_2984_pats_22_vars_relapse_ext_pats',
                            'result_cat':'RELAPSE'},

                          'tb21_22_2984_pats_22_vars_relapse_without_dr_reg_ext_pats':{
                             'fn':'tb21_22_2984_pats_22_vars_result_at_end_of_treatment',
                            'pat_ids_fn':'tb21_22_2984_pats_22_vars_relapse_ext_pats',
                            'result_cat':'RELAPSE'},
                        
                        'tb21_22_18_3079_pats_22_vars_result_at_end_of_treatment':{
                            'result_cat':'RESULT_AT_END_OF_TREATMENT',
                            'selection_method':'patient_clustering',
                            'clust_comb':'2-3-4-5',
                            'graph_metric':None,
                            'num_of_common_vars':22,
                            'training_days':120}}

## Define string descriptions to add as a prefix for each ds_type, for LLM to know what the variables describe
ds_type_descriptions={  'dm':'Demographic descriptors',
                        'mb':'Microbiological test results',
                        'vs':'Vital signs',
                        're':'Chest X-ray findings',
                        'lb':'Laboratory test results',
                        'dr_reg':'Cumulative drug doses taken',
                        'mh':'Medical history',
                        'ms':'Microbiological susceptibility',
                        'cmdos':'Cumulative concomitant medication taken',
                        'ce':'Clinical events',
                        'su':'Substance use',
                        'ae':'Adverse events'}


##### ============== HELPER FUNCTIONS ===================

def get_all_var_names():
    f = open('../../data/all_pat_variables_with_reliable_therapy_data_dict',"rb")
    d=pickle.load(f)

    vars_per_dataset={}
    
    all_vars=set()
    for ds_type in [*d]:
        l=[]
        for pat in d[ds_type]:
            l.append(d[ds_type][pat])    
        
        unique_elements = set()
    
        # Loop through each sub-list in the ragged list
        for sublist in l:
            # Add each element to the set
            unique_elements.update(sublist)

        ## Update the dict subsetted per dataset type
        vars_per_dataset[ds_type]=unique_elements
        
        ## Update set containing all variable names
        all_vars.update(unique_elements)
        #print(unique_elements)    
    
    return vars_per_dataset,list(all_vars)

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






##### ============== Load prediction label dataframe and patient IDS ===================
all_vars_dict,all_vars=get_all_var_names()

outcome_df=pd.read_csv('../../data/tb_1018_20_21_22_30_outcome.csv.gz',index_col=0)
outcome_df=outcome_df.set_index('USUBJID',drop=True)


## LOOP THROUGH THE DIFFERENT DATASETS AND ALL THE DIFFERENT DATA LAYERS, AND CONVERT TABULAR DATA INTO DICTIONARY OF STRINGS
for key in [*parameters_for_analysis][:1]:
    ## Load data
    fn='../../data/'+key+'_all_data_concat.csv.gz'
    data_conc=pd.read_csv(fn,low_memory=False,index_col=0,usecols=range(5))
    pat_ids=data_conc['USUBJID'].unique().tolist()
    del data_conc
    

    ## Extract outcome label and convert FAVOURABLE/UNFAVOURABLE to 0/1
    outcome_label=key.split('vars_')[-1].upper()
    #outcome_df[outcome_label]=outcome_df[outcome_label].replace({'FAVOURABLE':1,'UNFAVOURABLE':0},regex=False).values
    target_df=outcome_df.loc[pat_ids,outcome_label]
    print(target_df.value_counts(dropna=False))
    print(key)







##### ============== LLM LOADER FUNCTIONS ===================
### -==============================================
def load_input_texts_and_labels(ds_types_merged,dataset_param_key,target_df,prompt,period_end_day,data_inclusion_type):
    if data_inclusion_type=='baseline_vars_ext':
        data_inclusion_type='all_days'
        
    if ds_types_merged==True:
        
        #fn=f'../../data/{dataset_param_key}_input_dict_all_ds_types_merged.json'
        fn=f'../../data/{dataset_param_key}_{period_end_day}_days_input_dict_all_ds_types_merged.json'
        fn=f'../../data/{dataset_param_key}_{period_end_day}_days_{data_inclusion_type}_input_dict_all_ds_types_merged.json'
        input_dict=json.load(open(fn))
        
        all_labels=target_df[[*input_dict]].values.tolist() #.astype(int)
        all_texts=np.array(list(input_dict.values())).tolist()
        all_texts=[f'{prompt} {text}' for text in all_texts]
        
        pat_ids=list(input_dict.keys())
        
        return pat_ids,None,all_texts,all_labels
        
    if ds_types_merged==False: 
        #fn=f'../../data/{dataset_param_key}_input_dict_ds_types_seperate.json'
        fn=f'../../data/{dataset_param_key}_{period_end_day}_days_input_dict_ds_types_seperate.json'
        fn=f'../../data/{dataset_param_key}_{period_end_day}_days_{data_inclusion_type}_input_dict_ds_types_seperate.json'
        input_dict=json.load(open(fn))

  
        
        all_ds_types=list(input_dict[[*input_dict][0]].keys())
        pat_ids=[*input_dict][:]
        
        all_texts={}
        
        for ds_type in all_ds_types[:]:
        
            all_texts[ds_type]={}
            pats_with_data=[]
            for pat_id in pat_ids:
                
                if len(input_dict[pat_id].keys())==len(all_ds_types):
                    pats_with_data.append(pat_id)
            
            all_text_per_ds_type = [(pat_id,input_dict[pat_id][ds_type]) for pat_id in pats_with_data]  
            all_text_per_ds_type=[f'{prompt} {text}' for text in all_text_per_ds_type]
            
            all_labels_per_ds_type = target_df.loc[pats_with_data].values.tolist() 
            
            all_texts[ds_type]['pat_ids']=pats_with_data
            all_texts[ds_type]['all_texts']=all_text_per_ds_type
            all_texts[ds_type]['all_labels']=all_labels_per_ds_type
            
        return None,[*all_texts],all_texts,None

### -==============================================
def load_huggingface_model(model_name,device,id2label,label2id,model_type):
    from transformers import AutoModelForSequenceClassification, BitsAndBytesConfig,AutoModelForCausalLM

    quant_config=BitsAndBytesConfig(
                load_in_4bit=True,
                load_in_8bit=False,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=False
            )


    from huggingface_hub import login
    
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
    cache_dir='../../huggingface_cache'


    if model_type=='AutoModelForSequenceClassification':
        model=AutoModelForSequenceClassification.from_pretrained(model_name,cache_dir=cache_dir,
                                                                 output_hidden_states=True,
                                                                 torch_dtype=torch.float16,                                                        
                                                                 num_labels=len(id2label.keys()),
                                                                 id2label=id2label, 
                                                                 label2id=label2id,
                                                                 device_map='auto',
                                                                 #quantization_config=quant_config,
                                                                 pad_token_id=2)

    if model_type=='AutoModelForCausalLM':
        model=AutoModelForCausalLM.from_pretrained(model_name,cache_dir=cache_dir,
                                                                 output_hidden_states=True,
                                                                 torch_dtype=torch.float16,                                                        
                                                                 #num_labels=len(id2label.keys()),
                                                                 #id2label=id2label, 
                                                                 #label2id=label2id,
                                                                 device_map='auto',
                                                                 quantization_config=quant_config)
    return model

### -==============================================
def load_huggingface_tokenizer(model_name):
    
    
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
        
    cache_dir='../../huggingface_cache'
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name,cache_dir=cache_dir)
    
    return tokenizer





######## ================ SETUP DEVICE & LLM MODEL TYPE ================

## SET UP DEVICE AND LOAD PRETRAINED MODEL FROM HUGGINGFACE
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')


## Select Model type:
# - AutoModelForSequenceClassification: MLP head on top of pretrained model, num of outputs==len(id2label) with Cross-Entropy label prediction
# -AutoModelForCausalLM: output is text generation
model_type='AutoModelForSequenceClassification'
#model_type='AutoModelForCausalLM'

## Set up prediction labels
id2label={0: "FAVOURABLE", 1: "UNFAVOURABLE"}
label2id={"FAVOURABLE": 0, "UNFAVOURABLE": 1}

## if necessary convert the labels to integers
if model_type=='AutoModelForSequenceClassification':
    target_df=target_df.replace(label2id,regex=False)

### PARAMETERS FOR SUBSETTING TRAINING DATA
ds_types_merged=True
#dataset_param_key=key







######## ================ LOAD LLM MODEL ================

from peft import LoraConfig, TaskType,get_peft_model


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
        #peft_model_path=f"../../data/{llm_model_name}_{fine_tuned_tag}_num_of_epochs_{num_of_epochs}_{period_end_day}_days"
        peft_model_path=f"../../data/{llm_model_name}_{fine_tuned_tag}_num_of_epochs_{num_of_epochs}"#_{period_end_day}_days"
        
        if fine_tuned_tag=='fine_tuned':
            model=PeftModel.from_pretrained(model,peft_model_path)
            model=model.merge_and_unload()
            
            print(f'{llm_model_name}_{fine_tuned_tag} PEFT fine-tuned adapter added')




######## ================ INIT TOKENISER & PROMPT FUNCTION ================

## LOAD TOKENIZER AND TOKENIZE INPUT TEXTS
#tokenizer=load_huggingface_tokenizer(model_name)
tokenizer = AutoTokenizer.from_pretrained(llm_model_name,
                                          #force_download=False,
                                            #truncation_side="left",
                                            #padding_side="right",
                                            add_eos_token=True,
                                            add_bos_token=True
                                         )


### -=========================
## DEFINE TIMEPOINT OF PREDICTION AND ADD IT IN A PROMPT TO THE BEGINNING OF EACH INPUT
def create_prompt(timepoint,data_inclusion_type):
    timepoint=outcome_label.replace('_',' ').lower().split(' at')[-1]
    if timepoint==' end of treatment':
        timepoint='6 months'

    #if data_inclusion_type=='baseline_last_day':
    #prompt=f'[INST] The following data originates from a patient with pulmonary tuberculosis, participating in a Phase 3 clinical trial. Please predict the outcome of the therapy {timepoint} after therapy induction as FAVOURABLE or UNFAVOURABLE. [/INST]'
    prompt=f'[INST] The following data originates from a patient with pulmonary tuberculosis, participating in a Phase 3 clinical trial. Please summarise the condition of the patient. [/INST]'

    return prompt
        
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id







######## ================ EXTRACTING EMBEDDING ================

from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from numpy import save
import time
model.eval()

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True




for dataset_param_key in dataset_name_:
    print('Extracting embeddings for dataset',dataset_param_key)

    ### PARAMETERS FOR SUBSETTING TRAINING DATA
    ds_types_merged=True
    #dataset_param_key=key
    
    #period_end_days=['baseline',31,62,93,125,160,'all']
    #data_inclusion_types=['all_days','baseline_vars','baseline_last_day']#,'baseline_vars_ext']
    
    #period_end_day='all'
    
    ## DEFINE TIMEPOINT OF PREDICTION AND ADD IT IN A PROMPT TO THE BEGINNING OF EACH INPUT
    timepoint=outcome_label.replace('_',' ').lower().split(' at')[-1]
    if timepoint==' end of treatment':
        timepoint='6 months'
    
    
    
    start=time.time() 
    ## IF ALL DATASET TYPES (md,mb, lb, re,...) ARE MERGED INTO ONE LONG SENTENCE ==> EACH ITEM IN THE "ALL_TEXTS" LIST CORRESPONDS TO ONE SENTENCE / PATIENT
    #  ==> GET THE EMBEDDING FOR EACH PATIENT BY TOKENISING & FEEDING THE SENTENCE TO THE MODEL
    
    if ds_types_merged==True:
        
    
        for data_inclusion_type in data_inclusion_types_[::-1]:
          
            for period_end_day in period_end_days_[::-1]:
    
        
                prompt=create_prompt(timepoint,data_inclusion_type)
            
                pat_ids,all_ds_types,all_texts,all_labels = load_input_texts_and_labels(ds_types_merged,
                                                                                    dataset_param_key,
                                                                                    target_df,
                                                                                    prompt,
                                                                                    period_end_day,
                                                                                    data_inclusion_type)
            
            
      
                model_id=f"{llm_model_name}_{fine_tuned_tag}"
                #output_dir=f"../../data/{dataset_param_key}_{model_id}_{period_end_day}_days_{data_inclusion_type}"
                #os.makedirs(output_dir,exist_ok=True)
                
                ## TOKENIZE DATASET
                all_dataset_raw=Dataset.from_dict({"text": all_texts, 
                                                   #"labels": all_labels[:],
                                                   'pat_ids':[*pat_ids]}).with_format("torch")
                all_dataset_tokenized=all_dataset_raw.map(lambda x: tokenizer(x['text'],truncation=False, padding=False, return_tensors='pt'),batched=False)
                all_loader=DataLoader(all_dataset_tokenized, batch_size=1)
                
            
                
                
                with torch.no_grad():

                    for pool_method in pool_methods_:
                        
                        
                        output_dir=f"../../data/{dataset_param_key}_{model_id}_{period_end_day}_days_{data_inclusion_type}_{pool_method}"
                        os.makedirs(output_dir,exist_ok=True)
                        
                        #print(' - '.join([model_id,dataset_param_key,data_inclusion_type,period_end_day,pool_method]))
                        print(f'{model_id} - {dataset_param_key} - {data_inclusion_type} - {period_end_day} - {pool_method}')
                        
                        #print(model_id,dataset_param_key,period_end_day)
                        for n, batch in enumerate(tqdm(all_loader, desc="Processing", unit="batch")):
                            input_ids, attention_mask, pat_id = batch['input_ids'],batch['attention_mask'],batch['pat_ids']
                            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
        
                            #print('input_ids.shape',input_ids.shape)
                            
                            ## Get mean of embedding vector rowvise ==> mean(input_token_dim x embed_dimension) ==> embed_dimension
                            
                            #mask=attention_mask.detach().cpu()!=0
                            
                            if 'long-t5' in llm_model_name:
                                outputs = model(input_ids, attention_mask=attention_mask)
                                embed=outputs.last_hidden_state
                                #embed=embed[mask,:].detach().cpu()
                                embed_mean=torch.mean(embed[mask,:].detach().cpu(),axis=0)
                                fn=f"{output_dir}/{pat_ids[0].replace('/','_')}.pt"
                                torch.save(embed_mean,fn)
                            
                            else:
                                if 'TB-1018' in pat_id[0]:
                                    #print(pat_id)
                                    continue
                                  
                                if 'TB-1018' not in pat_id[0]:

                                    if pool_method=='mean_pooling':
                                        outputs = model(input_ids.squeeze(0), 
                                                        attention_mask=attention_mask,
                                                        use_cache=False, #labels=labels,
                                                        output_attentions=False)
                                        
                                        del input_ids,attention_mask
                                        embed=(outputs.hidden_states[-1]).squeeze(0)
                                        
                                        del outputs
                                        torch.cuda.empty_cache()
                                        
                                        embed_mean=torch.mean(embed.detach(),axis=0)
                                        
                                        del embed
                                        torch.cuda.empty_cache()
                                        
                                        fn=f"{output_dir}/{pat_id[0].replace('/','_')}.pt"
                                        torch.save(embed_mean,fn)
                                        
                                        del embed_mean
                                        torch.cuda.empty_cache()

                                    
                                    
                                    
                                    if pool_method=='attention_pooling':
                                    
                                        outputs = model(input_ids.squeeze(0), 
                                                        attention_mask=attention_mask,
                                                        use_cache=False, #labels=labels,
                                                        output_attentions=True)
                                        
                                        
                                        # ---- 2) Last 4 attention layers with minimal memory ----
                                        # outputs.attentions is a tuple(list) of length num_layers
                                        # each element: [num_heads, seq, seq] for decoder-only models (or [batch, heads, seq, seq] depending on architecture)
                                        #attentions =   # tuple of length L
                
                                        # Get only the last 4 layers
                                        #last4_attn = attentions[-4:]  # list of 4 tensors
                
                                        # Instead of saving full [layers x heads x seq x seq], we
                                        # reduce them to per-token importance scores to save memory.
            
                                        token_importance = None
                                        n_layers = 0
                
                                        for att in outputs.attentions[-4:]:
                                            # att: [batch, heads, seq, seq] -> [heads, seq, seq]
                                            if att.dim() == 4:
                                                att = att.squeeze(0)
                                        
                                            # mean over heads and query positions -> [seq_key]
                                            per_layer_importance = att.mean(dim=(0, 1))#.detach().cpu()  # [seq]
                                            del att
                                            #torch.cuda.empty_cache()
                                        
                                            if token_importance is None:
                                                token_importance = per_layer_importance
                                            else:
                                                token_importance = token_importance + per_layer_importance
                                        
                                            n_layers += 1
                                            # free this layer's attention tensor
                                            del per_layer_importance
                                            #torch.cuda.empty_cache()
                                        
                                        embed=(outputs.hidden_states[-1]).squeeze(0)
                                    
                                        del outputs
                                        #torch.cuda.empty_cache()
                                        
                                        # average over the 4 layers
                                        token_importance = token_importance / n_layers  # [seq]
                                        
                                        # Apply padding mask and normalise
                                        #token_importance = token_importance * mask.float()
                                        if token_importance.sum() > 0:
                                            token_importance = token_importance / token_importance.sum()
                
                                        
                                        #token_importance_cpu = token_importance.detach().cpu()
                                        weighted_embed = (embed * token_importance.unsqueeze(-1)).sum(dim=0)#.cpu()
    
                                        del embed
                                        
                                        fn=f"{output_dir}/{pat_id[0].replace('/','_')}.pt"
                                        torch.save(weighted_embed,fn)
    
                                        del weighted_embed
                                        #torch.cuda.empty_cache()
    
                                        fn=f"{output_dir}/{pat_id[0].replace('/','_')}_tok_imp.pt"
                                        torch.save(token_importance,fn)
                                        
                                        del token_importance                                   
                                        torch.cuda.empty_cache()
                                        
