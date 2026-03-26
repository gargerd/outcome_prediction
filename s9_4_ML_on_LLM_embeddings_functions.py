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



parameters_for_analysis={'tb21_22_2984_pats_22_vars_result_at_end_of_treatment':{
                            'fn':'tb21_22_2984_pats_22_vars_result_at_end_of_treatment',
                            'result_cat':'RESULT_AT_END_OF_TREATMENT'},
            
                        'tb21_22_2984_pats_22_vars_relapse':{
                            'fn':'tb21_22_2984_pats_22_vars_result_at_end_of_treatment',
                            'result_cat':'RELAPSE'}, 

                        'tb20_21_22_2908_pats_7_vars_relapse':{
                            'result_cat':'RELAPSE',
                            'fn':'tb20_21_22_2908_pats_7_vars_relapse',
                           'include_rifaquin':True},


                         
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
## Define string descriptions to add as a prefix for each ds_type, for LLM to know what the variables describe
ds_type_descriptions={'dm':'Demographic descriptors',
                    'mb':'Microbiological test results',
                    'vs':'Vital signs',
                    're':'Chest X-ray findings',
                    'lb':'Laboratory test results',
                    'dr_reg':'Cumulative drug doses taken'}

outcome_df=pd.read_csv('../data/tb_1018_20_21_22_30_outcome.csv.gz',index_col=0)
outcome_df=outcome_df.set_index('USUBJID',drop=True)

therapy_day_thr=80

## Set up prediction labels
id2label={0: "FAVOURABLE", 1: "UNFAVOURABLE"}
label2id={"FAVOURABLE": 0, "UNFAVOURABLE": 1}

### EXTRACT METADATA OF PATIENTS AND ADD THEM AS COLORING VARIABLES TO THE UMAP PLOT
## Define columns to use for UMAP colouring
col_columns=['STUDYID','AGE','SEX','RACE','ARM']

## Drop patients who have their last data at an earlier timepin than threshold
therapy_day_thr=80

llm_model_names=['BioMistral/BioMistral-7B',"epfl-llm/meditron-70b","epfl-llm/meditron-7b",
                 "google/long-t5-local-base",'google/long-t5-tglobal-large',
                'text-embedding-3-small']

fine_tuned_tags=['base','fine_tuned']
training_data_types=['full','pca']
pca_dims=50
model_names=['XGBoost','LogisticRegression',]
period_end_days=['baseline',31,62,93,125,160,'all']



##=========================================
## Setup function for calculating elapsed time
def print_elapsed_time(start,stop):
    # Calculate the elapsed time in seconds
    elapsed_seconds = stop - start
    
    # Convert elapsed time to hours and minutes
    elapsed_minutes, elapsed_seconds = divmod(int(elapsed_seconds), 60)
    elapsed_hours, elapsed_minutes = divmod(elapsed_minutes, 60)
    
    # Print the result in the desired format
    print(f"Elapsed time:{elapsed_hours} hours:{elapsed_minutes} minutes seconds:{elapsed_seconds}")


##=========================================
##  LOAD INPUT EMBEDDINGS CREATED BY AN LLM MODEL AS A DATAFRAME
'''
def load_input_emebddings_of_model(data_param_key,llm_model_name,fine_tuned_tag,period_end_day,data_inclusion_type,autoencoder_merged):

    if autoencoder_merged==True:
        model_dir=f"../data/{'_'.join([parameters_for_analysis[data_param_key]['fn'],llm_model_name,fine_tuned_tag,str(period_end_day),'days',data_inclusion_type,'autoencoder_merged'])}"
        fn=f"{model_dir}/latent_embeddings.csv.gz"
        df_pt=pd.read_csv(fn,low_memory=False,index_col=0)

        return df_pt

    
    if autoencoder_merged==False:
        
        ## LOAD EMBEDDINGS OF GIVEN MODEL AND DATASET&PREDICTION LABEL (CONTAINED IN 'key' string)
        model_dir=f"../data/{'_'.join([parameters_for_analysis[data_param_key]['fn'],llm_model_name,fine_tuned_tag,str(period_end_day),'days',data_inclusion_type])}"
        print(model_dir)  

        if 'text-embedding' in llm_model_name:
            llm_model_name_with_tag=llm_model_name
        else:
            llm_model_name_with_tag='_'.join([llm_model_name.split('/')[-1],fine_tuned_tag])
    
    
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
'''
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



###=========================================
### Initialise Dense Network structure -> number of layers and activation function as hyperparameters
class DenseNetwork(nn.Module):
    def __init__(self, input_dim, 
                 output_dim, 
                 hidden_dims,
                 hidden_activation,
                 last_activation):
        super(DenseNetwork, self).__init__()
        
        dims=[input_dim] + hidden_dims + [output_dim]
        layers=[]
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(hidden_activation)
        
        # Replace the last activation function
        layers[-1]=last_activation  
        #layers=layers[:-1]
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

#=================================================================
## Initialise CustomDataset for loading my training/testing data
class CustomDataset(Dataset):
    def __init__(self, x_train, y_train):
        self.x_train = x_train.reset_index(drop=True)
        self.y_train = y_train.reset_index(drop=True)

    def __len__(self):
        return len(self.x_train)

    def __getitem__(self, index):
        x = self.x_train.loc[index,:]
        y = self.y_train.loc[index,:]
        X_patient_data=torch.tensor(x.values, dtype=torch.float32)
        y_patient_data=torch.tensor(y.values, dtype=torch.float32)
        return X_patient_data, y_patient_data

#=================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, weight=None, reduction='mean'):
        """
        gamma: focusing parameter (>=0). 0 -> plain CE
        alpha: None or tensor of shape [num_classes] for class balancing
        weight: like CrossEntropyLoss 'weight' (class weights)
        """
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

        # class weights passed to CE
        self.register_buffer('weight', None if weight is None else torch.as_tensor(weight, dtype=torch.float))

        # focal alpha (per-class)
        if alpha is not None:
            self.register_buffer('alpha', torch.as_tensor(alpha, dtype=torch.float))
        else:
            self.alpha = None

    def forward(self, logits, targets):
        """
        logits: (N, C)
        targets: (N,) with class indices
        """
        # standard CE per-sample (no reduction)
        ce_loss = F.cross_entropy(logits, targets, weight=self.weight, reduction='none')  # (N,)

        # pt = exp(-CE) = predicted prob of true class
        pt = torch.exp(-ce_loss)  # (N,)

        # focal factor
        focal_factor = (1 - pt) ** self.gamma  # (N,)

        if self.alpha is not None:
            # get alpha per sample: alpha_t = alpha[targets]
            alpha_t = self.alpha[targets]  # (N,)
            loss = alpha_t * focal_factor * ce_loss
        else:
            loss = focal_factor * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

#=================================================================
## Function for training DenseNetwork
def train_dense_network(X_train,
                        y_train_data_with_index,
                        dense_network_params,
                        label_weights_dict,
                        train_param_comb,
                        plot_loss_function,
                        verbose,
                        patience=10,
                        X_val=None,
                        y_val_data_with_index=None):
    
    from torch.optim.lr_scheduler import CosineAnnealingLR

    if 'early_stop_epoch' in dense_network_params.keys():
        early_stopping_by_predef_epoch=True
        stop_epoch_num=int(np.ceil(dense_network_params['early_stop_epoch'])) + 1
    
    if 'early_stop_epoch' not in dense_network_params.keys():
        early_stopping_by_predef_epoch=False
        

    if X_val is not None and y_val_data_with_index is not None:
        early_stopping_by_val_loss=True
    else:
        early_stopping_by_val_loss=False        

    
    ### TRAIN MODEL ###
    loss_items_list=[]
    # Define batch size
    batch_size=dense_network_params['batch_size']

    # Create data loaders
    train_dataset=CustomDataset(X_train,y_train_data_with_index)
    train_loader=DataLoader(train_dataset,batch_size=batch_size, shuffle=True)

    if early_stopping_by_val_loss==True:
        val_dataset= CustomDataset(X_val, y_val_data_with_index)
        val_loader= DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize the model and optimizer
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=DenseNetwork(len(X_train.columns), dense_network_params['output_dim'], 
                        dense_network_params['hidden_dims'],
                        dense_network_params['hidden_activation'],
                        dense_network_params['last_activation']).to(device)
  

    #print(model)

    optimizer=torch.optim.Adam(model.parameters(), lr=dense_network_params['learning_rate'],
                                weight_decay=dense_network_params['weight_decay'])

    if label_weights_dict is not None:
        cross_entropy_weights=[label_weights_dict[key] for key in np.sort([*label_weights_dict])]
        #print('DenseNetwork weights',(cross_entropy_weights))

    ## Initialize loss function    
    if dense_network_params['criterion']=='CrossEntropyLoss':
        if label_weights_dict is not None:
            weight_tensor = torch.tensor(cross_entropy_weights, dtype=torch.float, device=device)
            criterion=nn.CrossEntropyLoss(weight=weight_tensor)
        if label_weights_dict is None:
            criterion=nn.CrossEntropyLoss()

    elif dense_network_params['criterion'] == 'FocalLoss':
        # build class weights from label_weights_dict if provided
        if label_weights_dict is not None:
            ce_weights = [label_weights_dict[key] for key in np.sort([*label_weights_dict])]
        else:
            ce_weights = None
        
        # focal alpha from params (optional)
        focal_alpha = dense_network_params.get('focal_alpha', None)
        gamma = dense_network_params.get('focal_gamma', 2.0)

        target_label_freq=y_train_data_with_index.value_counts(normalize=True)
        #print('target_label_freq',target_label_freq)
        focal_alpha = torch.tensor([target_label_freq.loc[0].values[0],
                                    target_label_freq.loc[1].values[0]])
        #print('target_label_freq',target_label_freq)
        #print('focal_alpha',focal_alpha)
    
        criterion = FocalLoss(
            gamma=gamma,
            alpha=focal_alpha,
            weight=ce_weights,
            reduction='mean'
        ).to(device)


    elif dense_network_params['criterion']=='MSELoss':
        criterion=nn.MSELoss()
  

    # Train the model
    num_of_epochs=dense_network_params['num_epochs']

     # Learning rate scheduler
    scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=dense_network_params['num_epochs'],  # The maximum number of epochs
        eta_min=0  # Minimum learning rate
    )



     # ---- early stopping state ----
    best_val_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0

    
 

    epoch_loss_list = []
    
    for epoch in range(num_of_epochs):
        model.train()
        
        epoch_losses = []
        #print('epoch:',epoch)
        for inputs, targets in train_loader:
            #print('inputs.shape',inputs.shape,'targets.shape',targets.shape)
            inputs = inputs.to(device)
            targets = targets.to(device)
        
            outputs = model(inputs)
            #print('outputs.shape',outputs,'targets.shape',targets)
            if dense_network_params['criterion'] in ['CrossEntropyLoss', 'FocalLoss']:
                loss = criterion(outputs, targets.squeeze(1).long())
            
            elif dense_network_params['criterion']=='MSELoss':
                loss = criterion(outputs, targets.squeeze(1))
            #print('loss',loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # store per-batch loss
            epoch_losses.append(loss.item())

            if plot_loss_function==True:
                loss_items_list.append(loss.item())
        '''
        if verbose==True:
            if num_of_epochs>10:
                if epoch==0:
                    print(f'Epoch [{epoch+1}/{num_of_epochs}], Loss: {loss.item():.3f}')
                if (epoch+1) % round(num_of_epochs/10) == 0:
                    print(f'Mean Loss Epoch [{epoch+1}/{num_of_epochs}], Loss: {round(np.mean(loss_items_list[-round(num_of_epochs/10):]),3)}')
            else:
                print(f'Epoch [{epoch+1}], Loss: {loss.item():.3f}') 
        '''

        mean_epoch_loss = np.mean(epoch_losses)
        epoch_loss_list.append(mean_epoch_loss)

        if early_stopping_by_val_loss==True:
            # -------- VALIDATION --------
            model.eval()
            val_losses = []
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs = inputs.to(device)
                    targets = targets.to(device)
    
                    outputs = model(inputs)
                    if dense_network_params['criterion'] in ['CrossEntropyLoss', 'FocalLoss']:
                        val_loss = criterion(outputs, targets.squeeze(1).long())
    
                    elif dense_network_params['criterion']=='MSELoss':
                        val_loss = criterion(outputs, targets.squeeze(1))
                   
                    val_losses.append(val_loss.item())
    
            mean_val_loss = float(np.mean(val_losses))
        
        
        scheduler.step()

        
        if verbose==True:
            if num_of_epochs > 10:
                #if epoch==0:
                #    print(f'Epoch [{epoch+1}/{num_of_epochs}], Loss: {loss.item():.3f}')
                    
                if (epoch + 1) % round(num_of_epochs / 10) == 0:
                    n = round(num_of_epochs / 10)
                    mean_last_n_epochs = np.mean(epoch_loss_list[-n:])
                    print(f"Mean Loss over last {n} epochs at epoch {epoch+1}: "
                          f"{mean_last_n_epochs:.3f}")
            else:
                print(f"Epoch [{epoch+1}], Loss: {mean_epoch_loss:.3f}")

        if early_stopping_by_val_loss==True:

            # -------- EARLY STOPPING CHECK --------
            
            if mean_val_loss < best_val_loss - 1e-6:  # small tolerance
                best_val_loss = mean_val_loss
                best_model_state = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    #if verbose:
                    print(f"Early stopping at epoch {epoch+1}. Best val loss: {best_val_loss:.3f}")
                    break
            
        # -------- EARLY STOPPING AT PREDEFINED EPOCH --------
        if early_stopping_by_predef_epoch==True and epoch >= stop_epoch_num:
            print(f"Early stopping at predefined epoch {epoch+1}.")
            best_model_state = copy.deepcopy(model.state_dict())
            break
                

    # restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        
    
    ### PLOT LOSS FUNCTION OVER TRAINING STEPS
    if plot_loss_function==True:
        fig,ax=plt.subplots(1,1,figsize=(10,5))

        sns.lineplot(y=loss_items_list,x=np.arange(len(loss_items_list)),ax=ax)
        #ax.set_title(categorical_map_name,fontsize=10)
        ax.set_ylim(0,None)
        ax.set_ylabel('Loss',fontsize=7)
        fig.suptitle(train_param_comb,fontsize=6,fontweight='bold') 
            
    return model, epoch



#=================================================================
## Function for testing the model 
#  1. Test the model with training dataset
#  2. Test the model with unseen testing dataset
def test_dense_model(model,X_test,y_test_data_with_index):


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.to(device)
    model.eval()
    
    ### TEST MODEL ###
    with torch.no_grad():
    
        x_test=X_test.copy()
        y_data_with_index=y_test_data_with_index.copy()                                                         

        test_dataset=CustomDataset(x_test,y_data_with_index)
        test_loader = DataLoader(test_dataset, batch_size=x_test.shape[0], shuffle=False)#, collate_fn=train_dataset.collate_fn)
        
        predicted_output_list,target_list=[],[]
        for inputs, targets in test_loader:
        
            #print('test input shape',inputs.shape)
            #print('test targets',targets.detach().cpu().numpy().flatten())
            predicted_output=model(inputs)
            #print('predicted_output',predicted_output.detach().cpu().numpy().flatten())

            #predicted_output_list.append(torch.argmax(predicted_output.detach(),dim=1).numpy().flatten())
            predicted_output_list.append(predicted_output.detach().numpy().flatten())
            #target_list.append(targets.detach().cpu().numpy().flatten())

    #predicted_output_list=np.array(predicted_output_list)

    predicted_output_list = predicted_output.squeeze(0).detach().numpy()
    
    return predicted_output_list





####======================================
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf
from sklearn.neighbors import NearestNeighbors

# --------- Helpers

def _safe_2d(X):
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return X

def _sigmoid_like(x, scale=1.0):
    # monotone decreasing conf from distances; stable and bounded
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + (x / (scale + 1e-8)))

# --------- Mahalanobis with optional PCA

@dataclass
class OODMahalanobis:
    use_pca: bool = False               
    pca_var: float = 0.95               # keep components explaining this variance
    lambda_scale: float = 1.0           # controls steepness of exp/score mapping
    standardize: bool = False            # standardize before PCA/cov
    _scaler: StandardScaler = None
    _pca: PCA = None
    _mu: np.ndarray = None
    _prec: np.ndarray = None            # precision matrix (cov^-1)
    _dist_scale: float = None           # robust scaling of distances for score mapping

    def fit(self, X_train):
        X = _safe_2d(X_train)
        # Standardize
        if self.standardize:
            self._scaler = StandardScaler()
            X = self._scaler.fit_transform(X)
        # PCA (optional, helps with embeddings / collinearity)
        if self.use_pca:
            self._pca = PCA(n_components=self.pca_var, svd_solver="full", random_state=0)
            X = self._pca.fit_transform(X)
        # Shrinkage covariance (stable in high-D)
        lw = LedoitWolf().fit(X)
        cov = lw.covariance_
        # Mean & precision
        self._mu = X.mean(axis=0)
        # Precision via pseudo-inverse for safety
        self._prec = np.linalg.pinv(cov)
        # Calibrate a robust distance scale (median of distances)
        d = self._mahal_dist(X)
        self._dist_scale = np.median(d) + 1e-8
        return self

    def _transform_X(self, X):
        X = _safe_2d(X)
        if self.standardize and self._scaler is not None:
            X = self._scaler.transform(X)
        if self.use_pca and self._pca is not None:
            X = self._pca.transform(X)
        return X

    def _mahal_dist(self, X_trans):
        diff = X_trans - self._mu
        # d^2 = (x-mu)^T * prec * (x-mu)
        d2 = np.einsum("...i,ij,...j->...", diff, self._prec, diff, optimize=True)
        return np.sqrt(np.clip(d2, 0, None))

    def distances(self, X):
        X_t = self._transform_X(X)
        return self._mahal_dist(X_t)

    def confidence(self, X):
        # Higher distance -> lower confidence; map distances to (0,1]
        d = self.distances(X)
        # Two good options; uncomment your preference

        # Option A: exponential falloff
        # return np.exp(-self.lambda_scale * d / (self._dist_scale + 1e-8))

        # Option B: logistic-like (robust to tails)
        return _sigmoid_like(d, scale=self._dist_scale / max(self.lambda_scale, 1e-8))

# --------- kNN density

@dataclass
class OODKNN:
    k: int = 20
    leaf_size: int = 30
    metric: str = "euclidean"
    standardize: bool = True
    use_pca: bool = False
    pca_var: float = 0.95
    _scaler: StandardScaler = None
    _pca: PCA = None
    _nn: NearestNeighbors = None
    _dist_scale: float = None

    def fit(self, X_train):
        X = _safe_2d(X_train)
        if self.standardize:
            self._scaler = StandardScaler()
            X = self._scaler.fit_transform(X)
        if self.use_pca:
            self._pca = PCA(n_components=self.pca_var, svd_solver="full", random_state=0)
            X = self._pca.fit_transform(X)
        self._nn = NearestNeighbors(n_neighbors=min(self.k, len(X)), leaf_size=self.leaf_size,
                                    metric=self.metric, n_jobs=-1)
        self._nn.fit(X)
        # Calibrate distance scale by average kNN distance on training
        d = self.avg_knn_distance(X)
        self._dist_scale = np.median(d) + 1e-8
        return self

    def _transform_X(self, X):
        X = _safe_2d(X)
        if self.standardize and self._scaler is not None:
            X = self._scaler.transform(X)
        if self.use_pca and self._pca is not None:
            X = self._pca.transform(X)
        return X

    def avg_knn_distance(self, X_trans):
        # returns average distance to k nearest neighbors for each sample
        # kneighbors returns distances to k neighbors per row (including self if fitted on same set)
        dists, _ = self._nn.kneighbors(X_trans, n_neighbors=min(self.k, self._nn.n_neighbors))
        # If self is included (distance 0), average still behaves sensibly
        return dists.mean(axis=1)

    def density(self, X):
        X_t = self._transform_X(X)
        avg_d = self.avg_knn_distance(X_t)
        # density proxy (higher is denser)
        dens = 1.0 / (avg_d + 1e-8)
        return dens

    def confidence(self, X):
        # Convert density to [0,1] confidence, robust scaling by training median distance
        X_t = self._transform_X(X)
        avg_d = self.avg_knn_distance(X_t)
        # map distance to confidence (lower distance -> higher confidence)
        return _sigmoid_like(avg_d, scale=self._dist_scale)



###=========================================
### FUNCTION FOR TRAINING THE ML MODELS & CALCULATING CV ROC-AUC SCORES
def init_model(model_name,X_train,y_train_data,
                                k_folds,random_state,
                                outcome_label,
                                model_params,
                                weight_by_label_freq,
                                train_params,
                                dense_network_params):

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

            #print('label_weights_dict',rf_label_weight)

        if train_params['label_weights'] is None:
            rf_label_weight=None   
            gb_label_weight=None

        if dense_network_params['label_weights'] is not None:
            label_weights_dict={label:x for label,x in enumerate(dense_network_params['label_weights'])}
        if dense_network_params['label_weights'] is None:
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
        
        model=LogisticRegression(penalty='elasticnet',solver='saga',class_weight=label_weights,max_iter=100,**model_params)
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

    elif 'Dense' in model_name:
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model=DenseNetwork(len(X_train.columns), dense_network_params['output_dim'], 
                            dense_network_params['hidden_dims'],dense_network_params['last_activation'],
                            dense_network_params['last_activation'])
        label_weights=None
        
    return model,label_weights,label_weights_dict


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
    X_train[X_train.columns]=std_scaler.transform(X_train)
    X_test[X_test.columns]=std_scaler.transform(X_test)
    return (X_train, X_test)




######====================================
## 1. Calibrate model with best hyperparams + extract calibrated prediction probs
## 2. Calculate confidence statistice from calirbated prediction probs.

def calibrate_model_and_extract_confidence_metrics(model,
                                                   X_train,
                                                   X_test,
                                                   y_train,
                                                   outcome_label,
                                                   label_weights_dict,
                                                   cv_roc_auc_scores=None,
                                                   cv_splitter=None):
    import copy

    ## Calculate calibrated probabilities of trained best model
    from sklearn.calibration import CalibratedClassifierCV

    if cv_splitter is None and cv_roc_auc_scores is not None:
        ## Extract train and validation ids to perform calibration on the same train-test sets as the hyperparam search
        cv_splits=zip(cv_roc_auc_scores['inner_train_val_splits']['train_ids'],
                      cv_roc_auc_scores['inner_train_val_splits']['test_ids'])
    
    if cv_splitter is None and cv_roc_auc_scores is None:
        raise('Please provide either a CV-splitter object for the internal CV or cv_roc_auc_scores dictionary of the trained model!')
        
    if cv_splitter is not None:
        
        ## STRATIFY ON OUTCOME LABEL & STUDYID 
        ##. ==> WITHIN STUDY ROC-AUC CLAUCLATION IS POSSIBLE, AS THERE ALWAYS WILL BE AT LEAST ONE UNFAVOUR. LABEL FROM BOTH STUDIES IN THE TEST SET
        df_=y_train.reset_index().drop_duplicates(subset='USUBJID')#.set_index('USUBJID',drop=True)
        df_['STUDYID']=df_['USUBJID'].str.split('/',expand=True)[0].values
        df_=df_.set_index('USUBJID')
        y_for_strat=df_[outcome_label].astype(str) + "_" + df_['STUDYID']#.astype(str)
    
        pat_ids=df_.index
        n_of_classes=len(y_train[outcome_label].unique())
        cv_splits=cv_splitter.split(pat_ids, y_for_strat.loc[pat_ids])

    
        
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
    probs_calibrated = calibrated_model.predict_proba(X_test)[:, 1]
    probs_uncalibrated = model.predict_proba(X_test)[:, 1]

    
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

    conf_metrics.index=X_test.index

    ## Add out-of-distribution confidence variables (Mahalanobis and KNN confidence)
    #ood_conf_df=cv_roc_auc_scores['ood_conf']
    #conf_metrics.loc[ood_conf_df.index,ood_conf_df.columns.tolist()] = ood_conf_df.values
    
    return conf_metrics#,calibrated_model



##=========================================
def return_ood_metrics(X_train,X_test,use_pca):
     
    # CAlculate Mahalanobis confidence value: Sigmoid(Mahal. distance to mean of training distribution)
    ood_mahal = OODMahalanobis(use_pca=use_pca, standardize=False).fit(X_train)
    mahal_conf_val = ood_mahal.confidence(X_test)  # shape (n_val,)

    # Calculate KNN confidence value: Sigmoid((1/average distance) to its k nearest neighbours in the training data)
    ood_kbb = OODKNN(use_pca=use_pca, standardize=False).fit(X_train)
    knn_conf_val = ood_kbb.confidence(X_test)  # shape (n_val,)

    ## Aggregate the OOD confidence values into one dataframe
    ood_df=pd.DataFrame({'knn_conf_val':knn_conf_val,
                              'mahal_conf_val':mahal_conf_val},
                               index=X_test.index)

    return ood_df


##=========================================
def run_cv(X,
           y,
           k_folds,
           model_name,
           weight_by_label_freq,
           random_state,
           outcome_label,
           model_params,
           train_params,
           dense_network_params,
           train_param_comb,
           calibrate_model=False):

    
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score,r2_score
    from sklearn.model_selection import cross_validate
    import time
    pd.options.mode.chained_assignment = None
    
    
    pat_ids=y.index.get_level_values('USUBJID').unique()
    y_unique=y.reset_index().drop_duplicates(subset='USUBJID').set_index('USUBJID',drop=True)
    #print('y_unique',y_unique.loc[pat_ids,:])

    ## Update dens network default parameters with the parameters selected for given CV run
    updated_dense_network_params=copy.deepcopy(dense_network_params)
    for param in model_params.keys(): 
        if param in dense_network_params.keys():
            updated_dense_network_params[param]=model_params[param]
   
    
    # Initialize a StratifiedKFold splitter
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=random_state)

    # Initialize lists to store the evaluation scores
    train_roc_auc_scores_all,test_roc_auc_scores_all,early_stop_epochs=[],[],[]
    train_roc_auc_scores_per_study,test_roc_auc_scores_per_study=[],[]
    train_ids_,test_ids_,conf_metrics_list=[],[],[]
    
    cv_roc_auc_scores={}

    cv_roc_auc_scores['inner_train_val_splits']={}


    ## STRATIFY ON OUTCOME LABEL & STUDYID 
    ##. ==> WITHIN STUDY ROC-AUC CLAUCLATION IS POSSIBLE, AS THERE ALWAYS WILL BE AT LEAST ONE UNFAVOUR. LABEL FROM BOTH STUDIES IN THE TEST SET
    ## CLASSIFIER
    y_for_strat=y[outcome_label].astype(str) + "_" + y.index.get_level_values('STUDYID')#.astype(str)

    ## REGRESSION
    #y_for_strat=y.reset_index().set_index('USUBJID')['STUDYID']

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
        
        
        # Split the data
        #X_train_fold, X_test_fold = scale_by_training_data(X.loc[train_mask,:], X.loc[test_mask,:])
        X_train_fold, X_test_fold =X.loc[train_mask,:], X.loc[test_mask,:]
        y_train_fold,y_test_fold=y.loc[train_mask,:],y.loc[test_mask,:]


        
        ## Init model

        model,label_weights,label_weights_dict=init_model(model_name,X_train_fold,y_train_fold,
                                                 k_folds,random_state,outcome_label,
                                                 model_params,
                                                 weight_by_label_freq,
                                                 train_params,
                                                 updated_dense_network_params) 
        #start=time.time()
        early_stop_epoch=None
        
        ## Fit model
        if model_name in ['XGBoost','GradientBoost']:
            model.fit(X_train_fold, y_train_fold.values.ravel(),sample_weight=label_weights)
        elif model_name in ['RandomForest','LogisticRegression','SVC','KNN']:
            model.fit(X_train_fold, y_train_fold.values.ravel())
        elif model_name in ['Dense']:
            model,early_stop_epoch = train_dense_network(X_train_fold,
                                                y_train_fold,
                                                updated_dense_network_params,
                                                label_weights_dict,
                                                train_param_comb,
                                                X_val=X_test_fold,
                                                y_val_data_with_index=y_test_fold,
                                                plot_loss_function=False,
                                                verbose=False)
        #stop=time.time()
        #print_elapsed_time(start,stop)
        
        if 'Dense' in model_name:
            model.eval()  # Set the model to evaluation mode
            with torch.no_grad():  # Disable gradient calculations
                #inputs = torch.tensor(test_data, dtype=torch.float32)  # Your test data

                ## CLASSIFIER
                train_probabilities = test_dense_model(model,X_train_fold,y_train_fold)[:,1]
                #print('train_probabilities',test_dense_model(model,X_train_fold,y_train_fold))
                test_probabilities = test_dense_model(model,X_test_fold,y_test_fold)[:,1]

                ## REGRESSION
                #train_probabilities = test_dense_model(model,X_train_fold,y_train_fold)
                #test_probabilities = test_dense_model(model,X_test_fold,y_test_fold)
                '''
                fig,ax=plt.subplots(1,1,figsize=(10,5))
                #print(y_train_fold.values.flatten())
                #print(train_probabilities.flatten())
                sns.scatterplot(y=y_train_fold.values.flatten(),x=train_probabilities.flatten(),ax=ax,s=5)
                #ax.set_title(categorical_map_name,fontsize=10)
                ax.set_ylim(0,None)
                ax.set_ylabel('Loss',fontsize=7)
                fig.suptitle(train_param_comb,fontsize=6,fontweight='bold') 
                '''
                

        
        else:
            # Predict probabilities on the training and test data
            train_probabilities = model.predict_proba(X_train_fold)[:,1]
            test_probabilities = model.predict_proba(X_test_fold)[:,1]
        
        y_train_fold_=y_train_fold.copy()
        
        ## If patients come from multiple studies, calculate the ROC-AUC score within the studies as well
        if len(y.index.get_level_values('STUDYID').unique())>1:
            y_train_fold['pred']=train_probabilities
            y_test_fold['pred']=test_probabilities

            ## CLASSIFIER
            #train_roc_auc_per_study=y_train_fold.groupby('STUDYID').apply(lambda x:roc_auc_score(x[outcome_label],x['pred']))
            train_roc_auc_per_study = y_train_fold.groupby('STUDYID').apply(
                                                                            lambda x: roc_auc_score(x[outcome_label], x['pred'])
                                                                            if x[outcome_label].nunique() > 1 else np.nan
                                                                        )
            #print(y_test_fold.groupby('STUDYID').apply(lambda x:(x[outcome_label].value_counts())))
            
            #test_roc_auc_per_study=y_test_fold.groupby('STUDYID').apply(lambda x:roc_auc_score(x[outcome_label],x['pred']))
            test_roc_auc_per_study = y_test_fold.groupby('STUDYID').apply(
                                                                            lambda x: roc_auc_score(x[outcome_label], x['pred'])
                                                                            if x[outcome_label].nunique() > 1 else np.nan
                                                                        )

            ## REGRESSION
            #train_roc_auc_per_study=y_train_fold.groupby('STUDYID').apply(lambda x:r2_score(x[outcome_label],x['pred']))
            #print(y_test_fold.groupby('STUDYID').apply(lambda x:(x[outcome_label].value_counts())))
            #test_roc_auc_per_study=y_test_fold.groupby('STUDYID').apply(lambda x:r2_score(x[outcome_label],x['pred']))
            
            train_roc_auc_scores_per_study.append(train_roc_auc_per_study)
            test_roc_auc_scores_per_study.append(test_roc_auc_per_study)
            
      
        # Calculate ROC AUC scores for all prediction across all studies
        ## CLASSIFIER
        train_roc_auc=roc_auc_score(y_train_fold[outcome_label], train_probabilities)
        test_roc_auc=roc_auc_score(y_test_fold[outcome_label], test_probabilities)
        

        ## REGRESSION
        #train_roc_auc=r2_score(y_train_fold[outcome_label], train_probabilities)
        #test_roc_auc=r2_score(y_test_fold[outcome_label], test_probabilities)


        # Append the scores to the lists
        train_roc_auc_scores_all.append(train_roc_auc)
        test_roc_auc_scores_all.append(test_roc_auc)
        
        if model_name in ['Dense'] and early_stop_epoch is not None:
            early_stop_epochs.append(early_stop_epoch)


        if calibrate_model==True:
            use_pca=True
            conf_metrics = calibrate_model_and_extract_confidence_metrics(
                                                               model=model,
                                                               X_train=X_train_fold,
                                                               X_test=X_test_fold,
                                                               y_train=y_train_fold_,
                                                               outcome_label=outcome_label,
                                                               label_weights_dict=label_weights_dict,
                                                               cv_roc_auc_scores=None,
                                                               cv_splitter=skf)
   
            
            ood_df = return_ood_metrics(X_train_fold,X_test_fold,use_pca)
        
            #ood_conf_df=cv_roc_auc_scores['ood_conf']
            conf_metrics.loc[ood_df.index,ood_df.columns.tolist()] = ood_df.values
            conf_metrics_list.append(conf_metrics)
        

    ## Add train-test IDs
    cv_roc_auc_scores['inner_train_val_splits']['train_ids']=train_ids_
    cv_roc_auc_scores['inner_train_val_splits']['test_ids']=test_ids_

    if calibrate_model==True:
        ## Concatenate and add Confidence metrics + Mahalanobis and KNN confidence dataframes of the 5 validation sets
        cv_roc_auc_scores['inner_CV_conf_metrics'] = pd.concat(conf_metrics_list).loc[X.index,:]
    
    ## Add ROC-AUC values calculated across all studies to the result dictionary
    cv_roc_auc_scores['train_score']=train_roc_auc_scores_all
    cv_roc_auc_scores['test_score']=test_roc_auc_scores_all

    if len(early_stop_epochs)>0:
        cv_roc_auc_scores['early_stop_epochs']=early_stop_epochs

    print('CV test_score:',test_roc_auc_scores_all)
    print('CV train_score:',train_roc_auc_scores_all)

    ## If multiple studies, join the per-study ROC-AUC scores into one dataframe containing scores per CV-fold 
    if len(y.index.get_level_values('STUDYID').unique())>1:
        
        train_roc_auc_per_study_df=pd.concat(train_roc_auc_scores_per_study,axis=1)
        test_roc_auc_per_study_df=pd.concat(test_roc_auc_scores_per_study,axis=1)
        
        cv_roc_auc_scores['train_score_per_study']=train_roc_auc_per_study_df
        cv_roc_auc_scores['test_score_per_study']=test_roc_auc_per_study_df
    
    return cv_roc_auc_scores 





##=========================================
## RUN CV OF GIVEN MODEL & TRAIN FINAL MODEL AFTERWARDS
def run_parameter_search(model_name,
                         X_train,
                         y_train_data,
                         k_folds,
                         random_state,
                        outcome_label,
                        param_search_dict,
                        weight_by_label_freq,
                        train_params,
                        dense_network_params,
                        train_param_comb,
                        calibrate_model):
    
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
              
        ## CALCULATE CV-SCORES
        cv_roc_auc_scores=run_cv(X_train,
                                 y_train_data,
                                 k_folds,
                                 model_name,
                                 weight_by_label_freq,
                                 random_state,
                                 outcome_label,
                                 model_params,
                                 train_params,
                                 dense_network_params,
                                 train_param_comb,
                                 calibrate_model)

                               
        
        ## SAVE RESULT OF CV WITH GIVEN PARAMETER SET
        param_search_results[param_set_string]=cv_roc_auc_scores

        
    return param_search_results



##=========================================
def extract_best_model_params(param_search_results,metric_func,num_of_top_models_per_cv,model_name,
                              average_models_across_splits=False):
    import re
    
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

        if 'early_stop_epochs' in cv_rep_param_search_results[[*cv_rep_param_search_results][0]].keys():
            early_stopping=True
        else:
            early_stopping=False

        ## Construct a dataframe with the parmeter sets and the corresponding mean or median of the validation ROC-AUC of model trained with parameter set
        param_roc_auc_df=pd.DataFrame.from_dict({param_set_string:cv_rep_param_search_results[param_set_string]['test_score'] for param_set_string in cv_rep_param_search_results.keys()})
        best_model_params = np.mean(param_roc_auc_df,axis=0).sort_values(ascending=False).index[:num_of_top_models_per_cv]

    
        best_params_dict={}            
        for n,best_param_set in enumerate(best_model_params):
            
            ## Convert the best parameter-set to a dictionary
            ## Split at points, where '-' is followd by next "parameter_name:"

            if model_name=='Dense':
                pairs = re.split(r'-(?=[A-Za-z_]+:)', best_param_set)
                best_model_params = {
                    k: v for k, v in (pair.split(':', 1) for pair in pairs)}
            else:
                best_model_params={key_val_pair.split(':')[0]:key_val_pair.split(':')[1] for key_val_pair in best_param_set.split('-')}
            #best_model_params={key_val_pair.split(':')[0]:key_val_pair.split(':')[1] for key_val_pair in best_param_set.split('-')}

            if early_stopping==True:
                best_model_params['early_stop_epoch']=np.mean(cv_rep_param_search_results[best_param_set]['early_stop_epochs'])
                
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
def calc_roc_auc_score_of_model(model_name,
                                X_train,
                                y_train_data,
                                k_folds,
                                random_state,
                                outcome_label,
                                model_params,
                                weight_by_label_freq,
                                train_params,
                                dense_network_params,
                                train_param_comb,
                                calibrate_model):
        
    ### RUN CV & TRAIN MODEL AFTERWARDS
    if model_name in ['XGBoost','GradientBoost']:
        
        ## CALCULATE CV-SCORES
        cv_roc_auc_scores=run_cv(X_train,
                                 y_train_data,
                                 k_folds,
                                 model_name,
                                 weight_by_label_freq,
                                 random_state,
                                 outcome_label,
                                 model_params,
                                 train_params,
                                 dense_network_params,
                                 train_param_comb,
                                calibrate_model)
                    
        
        ## INITIALIZE MODEL & TRAIN 
        model,label_weights,label_weights_dict=init_model(model_name,X_train,y_train_data,
                                                          k_folds,random_state,outcome_label,
                                                          model_params,
                                                          weight_by_label_freq,
                                                          train_params,
                                                          dense_network_params) 

        X_train,_ = scale_by_training_data(X_train, X_train)
        model.fit(X_train, y_train_data,sample_weight=label_weights)
        
    #else:
    elif model_name in ['RandomForest','LogisticRegression','SVC','KNN']:
        
        ## CALCULATE CV-SCORES
        cv_roc_auc_scores=run_cv(X_train,
                                 y_train_data,
                                 k_folds,
                                 model_name,
                                 weight_by_label_freq,
                                 random_state,
                                 outcome_label,
                                 model_params,
                                 train_params,
                                 dense_network_params,
                                 train_param_comb,
                                 calibrate_model)
                          
        
        ## INITIALIZE MODEL & TRAIN 
        model,label_weights,label_weights_dict=init_model(model_name,X_train,y_train_data,
                                                          k_folds,random_state,outcome_label,
                                                          model_params,
                                                          weight_by_label_freq,
                                                          train_params,
                                                          dense_network_params) 
        X_train,_ = scale_by_training_data(X_train, X_train)
        model.fit(X_train, y_train_data)

    elif model_name in ['Dense']:

        ## CALCULATE CV-SCORES
        cv_roc_auc_scores=run_cv(X_train,
                                 y_train_data,
                                 k_folds,
                                 model_name,
                                 weight_by_label_freq,
                                 random_state,
                                 outcome_label,
                                 model_params,
                                 train_params,
                                 dense_network_params,
                                 train_param_comb,
                                 calibrate_model)

        ## INITIALIZE MODEL & TRAIN 

        ## Update dens network default parameters with the parameters selected for given CV run
        updated_dense_network_params=copy.deepcopy(dense_network_params)
        for param in model_params.keys(): 
            if param in dense_network_params.keys():
                updated_dense_network_params[param]=model_params[param]


        #print('before training\n')
        #print('model_params.keys()',model_params.keys())
        #print('\n')
        if 'early_stop_epoch' in model_params.keys():
            updated_dense_network_params['early_stop_epoch']=model_params['early_stop_epoch']
                
                
        model,label_weights,label_weights_dict=init_model(model_name,X_train,y_train_data,
                                                          k_folds,random_state,outcome_label,
                                                          model_params,
                                                          weight_by_label_freq,
                                                          train_params,
                                                          updated_dense_network_params) 

        model,_ = train_dense_network(X_train,
                                    y_train_data,
                                    updated_dense_network_params,
                                    label_weights_dict,
                                    train_param_comb,
                                    plot_loss_function=False,
                                    verbose=False)
        
    return model,cv_roc_auc_scores,label_weights_dict


##=========================================
### CALCLUATE PCA & STANDARD-SCALED DATA
def return_std_data_and_pca(df,num_pca_comp):
    
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    
    ## RUN PCA
    # Standardize the data
    scaler=StandardScaler()
    scaled_data=pd.DataFrame(data=scaler.fit_transform(df),index=df.index)

    ## Run PCA
    pca=PCA(n_components=num_pca_comp)
    pca.fit(scaled_data.T)   
    df_pca=pd.DataFrame(data=np.transpose(pca.components_),index=df.index)
    
    return scaled_data,df_pca

##=========================================
### CALCLUATE PCA & STANDARD-SCALED DATA
def return_pca(df,num_pca_comp):
    
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    
    ## RUN PCA
    # Standardize the data
    scaler=StandardScaler()
    scaled_data=pd.DataFrame(data=scaler.fit_transform(df),index=df.index)

    ## Run PCA
    pca=PCA(n_components=num_pca_comp)
    pca.fit(scaled_data.T)   
    df_pca=pd.DataFrame(data=np.transpose(pca.components_),index=df.index)
    
    return df_pca



    

    
##=========================================    
def draw_pca_biplot(score, coeff, y, loading_rel_length_thr, legend_title, labels=None):
    from adjustText import adjust_text
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    xs = score[:, 0]
    ys = score[:, 1]
    
    load_vect_lengths = np.sqrt(coeff[:, 0]**2 + coeff[:, 1]**2)
    norm_load_vect_lengths = load_vect_lengths / np.max(load_vect_lengths)

    coeff_filt = coeff[norm_load_vect_lengths > loading_rel_length_thr]
    #labels = [x.split('_STD_NUM_RESULT')[0] for x in labels]
    #labels = [x.split('dr_reg_')[-1] for x in labels]
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
    
    if temp_data_name!='cmdos':
        vars_per_pat=variables_per_patient_all.loc[X_subset['USUBJID'].unique(),:]
        a=vars_per_pat.loc[:,vars_per_pat.columns.str.startswith(temp_data_name)].sum()
        temp_cols_with_suff_data=a[a>len(X_subset['USUBJID'].unique())*temp_col_threshold].index.tolist()

    if temp_data_name=='cmdos':
        aa=X_subset.loc[:,['USUBJID']+ X_subset.columns[X_subset.columns.str.startswith('cmdos')].tolist()]
        bb=aa.groupby('USUBJID').apply(lambda x:pd.DataFrame(index=x.dropna(how='all',axis=1).columns,data=[1,]*x.dropna(how='all',axis=1).shape[1]).T)
        temp_cols_with_suff_data = bb.loc[:,bb.columns.str.contains('cumul')].sum()[bb.loc[:,bb.columns.str.contains('cumul')].sum()>10].index.tolist()


    return temp_cols_with_suff_data


##========================================= 
##========================================= 
### 1. COLLECT PATIENTS, WHO ONLY HAVE FAVOURABLE OUTCOMES AT END OF TREATMENT & AT ALL FOLLOW-UP TIMEPOINTS 
    #. ==>TB-1021: 12 & 18 MONTHS, TB-1022: 18 & 24 MONTHS

### 2. COLLECT PATIENTS, WHO HAVE FAVOURABLE OUTCOME AT END OF TREATMENT, BUT HAVE AT LEAST ONE UNFAVOURABLE OUTCOME AT ANY OF THE FOLLOW-UP TIMEPOINTS 
    #. ==>TB-1021: 12 & 18 MONTHS, TB-1022: 18 & 24 MONTHS

def extract_rifaquin_relapse():

    def load_merged_data_of_lab_vars():
        #load patient IDs who are considered in this  analysis
        pat_id_df=pd.read_csv('../data/patients_in_analysis.csv.gz',index_col=0)
        # get all pat ids
        all_ids=pat_id_df['USUBJID'].to_list()
    
        fname='merged_df.csv.gz'
        
        fn=os.path.join('../data/',fname)
        merged_df=pd.read_csv(fn,low_memory=False,index_col=0)
    
        return merged_df
    
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


    ## For patients with treatment gaps, extract the first day after the treatment gap
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
def subset_pats_with_therapy_in_period(period_num,period_end_days):#,last_init_ther_days,pats_with_relapse_df):
    
     ## SUBSET TO PATIENTS WHO WERE TAKING DRUGS DURING THE PERIOD
    if parameters_for_analysis[data_param_key]['result_cat']!='RESULT_AT_END_OF_TREATMENT': 
        
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
    if parameters_for_analysis[data_param_key]['result_cat']=='RELAPSE' or 'pred_prob' in parameters_for_analysis[data_param_key]['result_cat']: 
        
        if isinstance(period_end_day,int):#!='all':
            pat_ids_ = pats_with_relapse_df[(pats_with_relapse_df['RELAPSE_DAY']>period_end_day)\
                                      |(pats_with_relapse_df['RELAPSE_DAY'].isna())]['RELAPSE_DAY'].index.tolist() #
    
        if period_end_day=='all':
            pat_ids_ = pats_with_relapse_df[(pats_with_relapse_df['RELAPSE_DAY']>182)\
                                             |(pats_with_relapse_df['RELAPSE_DAY'].isna())]['RELAPSE_DAY'].index.tolist()

        if period_end_day=='baseline':
            pat_ids_ = pats_with_relapse_df[(pats_with_relapse_df['RELAPSE_DAY']>0)\
                                             |(pats_with_relapse_df['RELAPSE_DAY'].isna())]['RELAPSE_DAY'].index.tolist()

    return pat_ids_



#####================
def return_predict_label_dataframe(parameters_for_analysis,data_param_key,X,
                                  outcome_df,outcome_label,model_names):

    import copy
    
    ## If not RELAPSE shpuld be predicted, subset the patients according to the availbility of the outcome results
    if parameters_for_analysis[data_param_key]['result_cat']=='RESULT_AT_END_OF_TREATMENT':
        
        ## Extract patients who have their last therapy day before therapy_day_thr ==> these patient probably dropped out
        last_day_per_pat_df=X.sort_values(by=['DAY']).groupby('USUBJID').apply(lambda x: x.loc[x.index[-1],:])
        pat_ids=last_day_per_pat_df[last_day_per_pat_df['DAY']>therapy_day_thr]['USUBJID'].tolist()
        #pat_ids=X['USUBJID'].unique().tolist()
        
        ## Subset outcome dataframe to patient considered
        target_df=outcome_df.loc[:,outcome_label]
        y=target_df.replace(label2id)
    
        y=y.reset_index()
        y['STUDYID']=y['USUBJID'].str.split('/',expand=True)[0].values
        y=y.set_index(['USUBJID','STUDYID'])
        #y=y.rename(columns={'index':'USUBJID'})
        outcome_label_=copy.deepcopy(outcome_label)
        
    if parameters_for_analysis[data_param_key]['result_cat']=='RELAPSE':
        #outcome_label='RELAPSE'
        include_rifaquin = parameters_for_analysis[data_param_key].get('include_rifaquin',None)
        pats_with_relapse_df=extract_21_22_relapse_pats(include_rifaquin)
        #pats_with_relapse_df=extract_21_22_relapse_pats()
    
        ## IF PEDICTION IS A REGRESSION TASK
        #outcome_label='RELAPSE_DAY'
        pats_with_relapse_df = pats_with_relapse_df[[outcome_label]].dropna()
                                  
        pat_ids=pats_with_relapse_df.index.tolist()
        target_df=pats_with_relapse_df[[outcome_label]]
        y=pats_with_relapse_df[[outcome_label]]
       
    
        y=y.reset_index()
        y['STUDYID']=y['USUBJID'].str.split('/',expand=True)[0].values
        y=y.set_index(['USUBJID','STUDYID'])
        outcome_label_=copy.deepcopy(outcome_label)


    if 'pred_prob' in parameters_for_analysis[data_param_key]['result_cat']:

        ## Create list to collect dataframes of different pred loss clusters
        clust_df_list=[]
        
        for model_name in model_names[1:]:

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


                ## Load pred loss cluster
                if 'raw' in pred_prob_coln or 'diff' in pred_prob_coln:
                    data_dir_=f'../data/model_interpretation/baseline/pred_prob_clusters'
                    
                    training_data_type='last_therapy_day'

                    fn=os.path.join(data_dir_,
                                f'tb21_22_2984_pats_22_vars_relapse_days_{training_data_type}_{model_name}_{pred_prob_coln}_pred_prob_clusters.csv')
        
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
                clust_df_list.append(clust_df_[[f'{pred_prob_coln}_cluster_binary',f'{pred_prob_coln}_loss']])

        ## Add all pred_prb_loss clusters to one dataframe
        clust_df_concat = pd.concat(clust_df_list,axis=1)

        ## Select final prediction label
        outcome_label_ = f'{outcome_label}_cluster_binary'
        
        clust_df_concat=clust_df_concat[~clust_df_concat[outcome_label_].isna()]
        
        target_df=clust_df_concat[[outcome_label_]]
        y=clust_df_concat[[outcome_label_]]
        pat_ids=clust_df_concat.index.tolist()

        y=y.reset_index()
        y['STUDYID']=y['USUBJID'].str.split('/',expand=True)[0].values
        y=y.set_index(['USUBJID','STUDYID'])
        
        
    return pat_ids,y,target_df,outcome_label_


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
    
    return conf_metrics,calibrated_model