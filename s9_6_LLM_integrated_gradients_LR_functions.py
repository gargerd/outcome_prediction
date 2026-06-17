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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
## SCALE TRAINING AND TESTING DATA WITH A STANDARD SCALER TRAINED ON THE TRAINING DATA
def scale_by_training_data(X_train, X_test):
    from sklearn.preprocessing import StandardScaler
    '''
    ## Standardise non-binary numerical columns ==> select numerical columns and drop columns already scaled
    numerical_cols=X_train.select_dtypes(exclude=['object'])
    non_binary_num_cols = []

    for col in numerical_cols.columns:
        unique_values = np.sort(numerical_cols[col].unique())
        if np.array_equal(unique_values, np.array([0, 1]))==False:
            non_binary_num_cols.append(col)

    if len(non_binary_num_cols)>0:
        std_scaler = StandardScaler()
        std_scaler.fit(X_train.loc[:,non_binary_num_cols])
        X_train.loc[:,non_binary_num_cols]=std_scaler.transform(X_train.loc[:,non_binary_num_cols])
        X_test.loc[:,non_binary_num_cols]=std_scaler.transform(X_test.loc[:,non_binary_num_cols])

        #X_train.loc[:,non_binary_num_cols]=(StandardScaler().fit_transform(X_train.loc[:,non_binary_num_cols]))

    '''
    std_scaler = StandardScaler()
    std_scaler.fit(X_train)
    X_train.loc[:,:]=std_scaler.transform(X_train)
    X_test.loc[:,:]=std_scaler.transform(X_test)
    return (X_train, X_test)



##=========================================
##  LOAD INPUT EMBEDDINGS CREATED BY AN LLM MODEL AS A DATAFRAME
def load_input_emebddings_of_model(data_param_key,llm_model_name,fine_tuned_tag,period_end_day,llm_model_name_with_tag,data_inclusion_type,autoencoder_merged):

    if autoencoder_merged==True:
        model_dir=f"../data/{'_'.join([parameters_for_analysis[data_param_key]['fn'],llm_model_name,fine_tuned_tag,str(period_end_day),'days',data_inclusion_type,'autoencoder_merged'])}"
        fn=f"{model_dir}/latent_embeddings.csv.gz"
        df_pt=pd.read_csv(fn,low_memory=False,index_col=0)

        return df_pt

    
    if autoencoder_merged==False:
        
        ## LOAD EMBEDDINGS OF GIVEN MODEL AND DATASET&PREDICTION LABEL (CONTAINED IN 'key' string)
        model_dir=f"../data/{'_'.join([parameters_for_analysis[data_param_key]['fn'],llm_model_name,fine_tuned_tag,str(period_end_day),'days',data_inclusion_type])}"
        print(model_dir)
    
        #if 'text-embedding' in llm_model_name:
        #    llm_model_name_with_tag=llm_model_name
        #else:
        #    llm_model_name_with_tag='_'.join([llm_model_name.split('/')[-1],fine_tuned_tag])
    
    
        ## LOOP THORUGH PAT _IDS AND LOAD THE EMBEDDINGS VECTORS INTO AN DATAFRAME
        l_pt,l_npy=[],[]
        #pats=target_df.index.tolist()
        
        pat_filenames = os.listdir(f"{model_dir}")
        
        pat_ids=[pat.replace('_','/').replace('.pt','') for pat in pat_filenames if 'TB-1018' not in pat]
    
        if 'text-embedding' in llm_model_name_with_tag:
            pat_ids=[pat.replace('_','/').replace('.npy','') for pat in pat_filenames if 'TB-1018' not in pat]
    
        for pat in pat_filenames:
            fn=f"{model_dir}/{pat}"
            
            if 'text-embedding' in llm_model_name_with_tag:
                embed=np.load(fn)
            
            else:
                embed=torch.load(fn)
                #print(torch.isinf(embed).any())
                #if torch.isinf(embed).any():
                    #torch.isnan(tensor).any().item()
                    #print('Inf in embedding',pat)
    
                embed=embed.detach().cpu().numpy()
            
            if 'TB-1018' not in pat:
                l_pt.append(embed)
            
            #fn=f"../data/{model_dir}/{pat_ids[0].replace('/','_')}_pca_mean.npy"
            #l_npy.append(np.load(fn))
    
        df_pt=pd.DataFrame(data=np.array(l_pt),index=pat_ids)
        #df_npy=pd.DataFrame(data=np.array(l_npy),index=pats)
    
    
        ## REPLACE -INF OR INF VALUES WITH THE MEAN OF THE GIVEN COLUMN
        # Replace infinite values with NaN
        df_pt=df_pt.replace([np.inf, -np.inf], np.nan)
        #df_npy.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Fill NaNs with respective column means
        for column in df_pt.columns[df_pt.isna().any()]:
            column_mean = df_pt[column].astype(float).mean(skipna=True)
            df_pt[column]=df_pt[column].fillna(column_mean, inplace=False)
    
        # Fill NaN values with mean of respective columns
        #df_pt=df_pt.fillna(df_pt.mean())
        #df_npy.fillna(df_npy.mean(), inplace=True)
    
        return df_pt #,df_npy



###=========================================
###=========================================
def load_input_embeddings_of_model(data_param_key, llm_model_name, fine_tuned_tag, period_end_day, 
                                   llm_model_name_with_tag,data_inclusion_type, pool_method,
                                   autoencoder_merged, max_workers=8):
    
    import os
    import numpy as np
    import pandas as pd
    import torch
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    base = "../data"
    dataset_name = parameters_for_analysis[data_param_key]['fn']

    parts = [dataset_name, llm_model_name, fine_tuned_tag, str(period_end_day), 'days', data_inclusion_type,pool_method]
    if autoencoder_merged:
        parts.append('autoencoder_merged')
    model_dir = os.path.join(base, '_'.join(parts))

    # Load full matrix directly if autoencoder was used
    if autoencoder_merged:
        df_pt = pd.read_csv(f"{model_dir}/latent_embeddings.csv.gz", index_col=0, low_memory=False)
        return df_pt

    text_embedding_mode = 'text-embedding' in llm_model_name
    #llm_model_name_with_tag = llm_model_name if text_embedding_mode else '_'.join([llm_model_name.split('/')[-1], fine_tuned_tag])

    def load_embedding(entry):
        fname = entry.name
        if 'TB-1018' in fname:
            return None

        path = os.path.join(model_dir, fname)

        try:
            if text_embedding_mode:
                embed = np.load(path)
            else:
                embed = torch.load(path,map_location=torch.device('cpu')).detach().cpu().numpy()

            # Replace inf before appending
            embed = np.where(np.isinf(embed), np.nan, embed)

            pat_id = fname.replace('_', '/').replace('.pt', '').replace('.npy', '')
            return (pat_id, embed)
        except Exception as e:
            print(f"Error loading {fname}: {e}")
            return None

    # Use ThreadPoolExecutor for I/O parallelism
    embeddings = []
    pat_ids = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(load_embedding, entry)
            for entry in os.scandir(model_dir)
            if entry.is_file() and (entry.name.endswith('.npy') or entry.name.endswith('.pt'))
        ]

        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                pid, emb = result
                pat_ids.append(pid)
                embeddings.append(emb)

    # Stack into DataFrame
    df_pt = pd.DataFrame(data=np.vstack(embeddings), index=pat_ids)

    df_pt = df_pt.astype(float)

    # Clean Inf/NaN
    df_pt = df_pt.replace([np.inf, -np.inf], np.nan)
    df_pt = df_pt.fillna(df_pt.mean(numeric_only=True))

    return df_pt




### -==============================================
def load_input_texts_and_labels(ds_types_merged,dataset_param_key,target_df,prompt,period_end_day,data_inclusion_type):
    if data_inclusion_type=='baseline_vars_ext':
        data_inclusion_type='all_days'

    #key=[*parameters_for_analysis][0]
        
    if ds_types_merged==True:
        
        #fn=f'../data/{dataset_param_key}_input_dict_all_ds_types_merged.json'
        fn=f'../data/{dataset_param_key}_{period_end_day}_days_input_dict_all_ds_types_merged.json'
        fn=f'../data/{dataset_param_key}_{period_end_day}_days_{data_inclusion_type}_input_dict_all_ds_types_merged.json'
        input_dict=json.load(open(fn))
        
        all_labels=target_df[[*input_dict]].values.tolist() #.astype(int)
        all_texts=np.array(list(input_dict.values())).tolist()
        all_texts=[f'{prompt} {text}' for text in all_texts]
        
        pat_ids=list(input_dict.keys())
        
        return pat_ids,None,all_texts,all_labels
        
    if ds_types_merged==False: 
        #fn=f'../data/{dataset_param_key}_input_dict_ds_types_seperate.json'
        fn=f'../data/{dataset_param_key}_{period_end_day}_days_input_dict_ds_types_seperate.json'
        fn=f'../data/{dataset_param_key}_{period_end_day}_days_{data_inclusion_type}_input_dict_ds_types_seperate.json'
        input_dict=json.load(open(fn))

        '''
        all_labels=target_df[[*input_dict]].values.tolist() #.astype(int)

        pat_ids=list(itertools.chain(*[[pat_id,]*len(input_dict[pat_id]) for pat_id in [*input_dict]]))
        all_labels=target_df.loc[pat_ids].values.tolist() #.astype(int)
        all_ds_types=[ds_type for pat_id in input_dict.keys() for ds_type in input_dict[pat_id].keys()]
        all_texts=[text for text_dict in input_dict.keys() for text in input_dict[text_dict].values()]
        all_texts=[f'{prompt} {text}' for text in all_texts]
        return pat_ids,all_ds_types,all_texts,all_labels
        '''
        
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
    login("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")
    cache_dir='../huggingface_cache'


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
    from huggingface_hub import login
    login("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")
    cache_dir='../huggingface_cache'
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name,cache_dir=cache_dir)

    return tokenizer




##### IG FUNCTIONS

import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm

class SklearnLRWrapper(nn.Module):
    def __init__(self, sklearn_model):
        super().__init__()
        self.weights = nn.Parameter(
            torch.tensor(sklearn_model.coef_, dtype=torch.float32), requires_grad=False
        )  # shape (1, 4096)
        self.bias = nn.Parameter(
            torch.tensor(sklearn_model.intercept_, dtype=torch.float32), requires_grad=False
        )  # shape (1,)

    def forward(self, x):
        logits = torch.matmul(x, self.weights.T) + self.bias  # shape (batch, 1)
        #probs = torch.sigmoid(logits)
        return logits.squeeze(1)  # (batch,)
        #return probs



####=================================================

from captum.attr import IntegratedGradients

# --- Chunking function ---
def chunk_token_ids_with_indices(input_ids, max_len=512, overlap=64):
    chunks, starts = [], []
    start = 0
    while start < len(input_ids):
        end = min(start + max_len, len(input_ids))
        chunks.append(input_ids[start:end])
        starts.append(start)
        if end == len(input_ids):
            break
        start += max_len - overlap
    return chunks, starts

####=================================================

# --- IG for one chunk ---
def run_ig_on_chunk(input_ids_chunk, model,tokenizer,n_steps,internal_batch_size, mean,scale):
    
    input_ids_chunk = input_ids_chunk.unsqueeze(0).to(device)
    input_embeds = model.get_input_embeddings()(input_ids_chunk).detach().clone()
    input_embeds.requires_grad_()

    baseline_text="[INST] The following data originates from a patient with pulmonary tuberculosis, participating in a Phase 3 clinical trial. Please summarise The condition of The patient. [/INST]"
    #baseline_text="[INST][/INST]"

    baseline_ids = tokenizer(baseline_text, return_tensors='pt', add_special_tokens=True).input_ids.to(device)
    baseline_embeds = model.get_input_embeddings()(baseline_ids)

    if input_embeds.shape[1] != baseline_embeds.shape[1]:
        pad_len = input_embeds.shape[1] - baseline_embeds.shape[1]
        baseline_embeds = torch.nn.functional.pad(baseline_embeds, (0, 0, 0, pad_len))

    '''
    def forward_func(embed_input):
        output = model(inputs_embeds=embed_input, output_hidden_states=True)
        hidden = output.hidden_states[-1]  # final hidden state
        print("hidden mean shape:", hidden.mean(dim=1).shape)
        print("embedding_dim index:", embedding_dim)
        print("selected shape:", hidden.mean(dim=1)[:, embedding_dim].shape)
        return hidden.mean(dim=1)[:, embedding_dim]

    '''
    
    def forward_func(embed_input,mean,scale):
        output = model(inputs_embeds=embed_input, output_hidden_states=True)
        hidden = output.hidden_states[-1]  # (1, seq_len, 4096)
        mean_embed = hidden.mean(dim=1)    # (1, 4096)

         # Standardize using the same parameters as the training scaler
        standardized_embed = (mean_embed - mean) / scale
    
        return lr_torch(standardized_embed)  # scalar: P(Y=1)



    ig = IntegratedGradients(forward_func)
    print('input_embeds.shape',input_embeds.shape)
    print('baseline_embeds.shape',baseline_embeds.shape)
    attributions = ig.attribute(
        inputs=input_embeds,
        baselines=baseline_embeds,
        return_convergence_delta=False,
        n_steps=n_steps,
        internal_batch_size=internal_batch_size,
        additional_forward_args=(mean,scale),
    )

    print("🔎 IG attribution shape before squeeze:", attributions.shape)
    print("After squeeze:", attributions.squeeze(0).shape)
    return attributions.squeeze(0).detach().cpu(), tokenizer.convert_ids_to_tokens(input_ids_chunk[0].cpu())




### -==============================================
## DEFINE TIMEPOINT OF PREDICTION AND ADD IT IN A PROMPT TO THE BEGINNING OF EACH INPUT
def create_prompt():
    #timepoint=outcome_label.replace('_',' ').lower().split(' at')[-1]
    #if timepoint==' end of treatment':
    #    timepoint='6 months'

    #if data_inclusion_type=='baseline_last_day':
    #prompt=f'[INST] The following data originates from a patient with pulmonary tuberculosis, participating in a Phase 3 clinical trial. Please predict the outcome of the therapy {timepoint} after therapy induction as FAVOURABLE or UNFAVOURABLE. [/INST]'
    prompt=f'[INST] The following data originates from a patient with pulmonary tuberculosis, participating in a Phase 3 clinical trial. Please summarise the condition of the patient. [/INST]'

    return prompt


####=================================================


##### Create Baseline preserving the instruction and the ds_type (or modality) descriptions, with their values deleted 
def build_baseline_text_preserve_structure(full_text):

    
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
    """
    Build a baseline text preserving:
      - Newlines and formatting
      - [INST] ... [/INST] section
      - Modality order
      - Empty brackets [] after each modality description

    Removes:
      - Only the data inside square brackets [ ... ]

    Special case:
      - For modality 'dr_reg', uses 'by patient:' instead of 'of patient:'
    """

    text_baseline = full_text

    for code, desc in ds_type_descriptions.items():
        # Determine connector phrase: "of patient" or "by patient"
        connector = "by patient" if code == "dr_reg" else "of patient"

        # Regex: match "<desc> of/by patient: [ ... ]"
        pattern = re.compile(
            rf"({re.escape(desc)} {connector}:\s*)\[.*?\]",
            flags=re.DOTALL
        )

        # Replace the bracket content with empty brackets []
        text_baseline = re.sub(pattern, r"\1[]", text_baseline)

    # Strip trailing spaces on each line but preserve line breaks
    text_baseline = "\n".join([line.rstrip() for line in text_baseline.splitlines()])

    return text_baseline




####=================================================
# --- IG for one chunk using LayerIntegratedGradients ---
def run_lig_on_chunk(input_ids_chunk, model,tokenizer,n_steps,internal_batch_size, mean,scale,lr_torch,input_text):

    from captum.attr import LayerIntegratedGradients
    import torch.nn.functional as F
    
    """
    input_ids_chunk: 1D torch tensor of token ids (seq_len,)
    model: your HF model (with output_hidden_states=True)
    Returns:
      attributions_chunk: torch tensor (seq_len, embed_dim)  -- attributions at embedding layer
      tokens: list of token strings for the chunk (length seq_len)
    """

    # make sure it's CPU->device and has batch dim
    input_ids_chunk = input_ids_chunk.to(device).unsqueeze(0)   # shape (1, seq_len)
    seq_len = input_ids_chunk.shape[1]

    # prepare baseline token ids from baseline_text and pad/truncate to seq_len
    baseline_text = build_baseline_text_preserve_structure(input_text)
    #baseline_text = "[INST] The following data originates from a patient with pulmonary tuberculosis, participating in a Phase 3 clinical trial. Please summarise The condition of The patient. [/INST]"
    baseline_enc = tokenizer(baseline_text, add_special_tokens=True, return_tensors="pt")["input_ids"][0].to(device)  # 1D
    # pad or truncate baseline token ids to match chunk length
    if baseline_enc.shape[0] < seq_len:
        pad_len = seq_len - baseline_enc.shape[0]
        baseline_ids = F.pad(baseline_enc, (0, pad_len), value=tokenizer.pad_token_id)  # right-pad with pad token id
    else:
        baseline_ids = baseline_enc[:seq_len]

    baseline_ids = baseline_ids.unsqueeze(0)  # shape (1, seq_len)

    # tokenizer tokens for mapping back to strings
    tokens = tokenizer.convert_ids_to_tokens(input_ids_chunk.squeeze(0).cpu())


    # define forward func that takes input_ids (batch) and returns scalar probability
    def forward_func(input_ids,mean,scale,lr_torch):
        # input_ids shape: (batch, seq_len)
        outputs = model(input_ids=input_ids, output_hidden_states=True)
        
        mean_embed = outputs.hidden_states[-1].mean(dim=1)             # (batch, hidden_dim)
        del outputs
        # standardize using saved scaler params (mean, scale) in torch tensors on device
        
        
        mean_=torch.reshape(mean, shape = mean_embed.shape)
        scale_=torch.reshape(scale, shape = mean_embed.shape)
        del mean, scale
        #print('within forward_func, mean_embed.shape,mean_.shape,scale_.shape',mean_embed.shape,mean_.shape,scale_.shape)
        
        standardized_embed = (mean_embed - mean_) / scale_
        del mean_embed,mean_,scale_
        # pass through saved LR wrapper (lr_torch) which returns probability in (batch,1) or (batch,)
        probs = lr_torch(standardized_embed)        # expect shape (batch, 1) or (batch,)
        del standardized_embed
        
        # ensure a scalar per example; return shape (batch,) or (batch,1)
        #if probs.ndim == 2 and probs.shape[1] == 1:
        #    probs = probs.squeeze(1)
        # return tensor shape (batch,) - LIG expects the forward to return a scalar per sample
        return probs

    # instantiate LIG: attribute layer is the embedding layer
    lig = LayerIntegratedGradients(forward_func, model.get_input_embeddings())

    # compute attributions: inputs and baselines are token ids (not embeddings)
    # internal_batch_size controls interpolation batching (helps memory)
    attributions = lig.attribute(
        inputs=input_ids_chunk,         # shape (1, seq_len)
        baselines=baseline_ids,         # shape (1, seq_len)
        n_steps=n_steps,
        internal_batch_size=internal_batch_size,
        additional_forward_args=(mean,scale,lr_torch),
        return_convergence_delta=False
    )
    # attributions shape from LIG: (1, seq_len, embed_dim)  (same shape as embedding layer outputs)
    #print("🔎 LIG raw attribution shape:", attributions.shape)

    # remove batch dim and convert to CPU
    attributions = attributions.squeeze(0).detach().cpu()  # (seq_len, embed_dim)

    return attributions,tokens

    
    
 
###======================
def load_final_patient_for_analysis(parameters_for_analysis,
                                   data_param_key,
                                   ):
    import copy

    
    ## LOAD FINAL PATIENT IDS FOR ANALYSIS, SAVED DURING PREPROCESSING OF THE BASELINE MODELS IN NOTEBOOK S9_3
    if 'pat_ids_fn' in parameters_for_analysis[data_param_key].keys():
        fn=f"../data/{parameters_for_analysis[data_param_key]['pat_ids_fn']}_final_pat_ids_for_analysis.pickle"
    else:  
        fn=f'../data/{data_param_key}_final_pat_ids_for_analysis.pickle'
    with open(fn, 'rb') as handle:
        final_pat_ids_for_analysis=pickle.load(handle)
    
    
    final_pat_ids_for_analysis_ = {}

    ## IF WE TRAIN ONLY SELECTED STUDIES AND TEST ON A HELD-OUT STUDIES, 
    #. - LOAD THE PATIENTS CONSIDERED FOR GIVEN TIME PERIOD USING THE BASE EXPERIMENT CASE INDICATED IN "pat_ids_fn"
    #. - SET 100% OF TRAIN SET PATIENTS AS TRAIN SET
    #. - SET 100% OF VALIDATION SET PATIENTS AS TEST SET
    #. - CHANGE THE CV NUMBER TO 1, AS THERE IS NOR SUBSAMPLING OF PATIENTS
    if 'train' in data_param_key:
        training_cohort = parameters_for_analysis[data_param_key]['training_cohort']
        validation_cohort = parameters_for_analysis[data_param_key]['validation_cohort']
    
        for period_end_day in [*final_pat_ids_for_analysis][:]:
            
            final_pat_ids_for_analysis_[period_end_day] = {}
            
            for cv_repeat_num in [*final_pat_ids_for_analysis[period_end_day]][:1]:
                
                final_pat_ids_for_analysis_[period_end_day][cv_repeat_num]={}
                
                X_train_pat_ids=final_pat_ids_for_analysis[period_end_day][cv_repeat_num]['X_train_ids']
                X_test_pat_ids=final_pat_ids_for_analysis[period_end_day][cv_repeat_num]['X_test_ids']
    
                all_pats = X_train_pat_ids + X_test_pat_ids            
        
                X_train_pat_ids_ = [id_ for id_ in all_pats if any(sub in id_ for sub in training_cohort)]
                X_test_pat_ids_ = [id_ for id_ in all_pats if any(sub in id_ for sub in validation_cohort)]
              
                
                final_pat_ids_for_analysis_[period_end_day][cv_repeat_num]['X_train_ids'] = X_train_pat_ids_
                final_pat_ids_for_analysis_[period_end_day][cv_repeat_num]['X_test_ids'] = X_test_pat_ids_
                
    else:
        final_pat_ids_for_analysis_ = copy.copy(final_pat_ids_for_analysis)

    return final_pat_ids_for_analysis_




    