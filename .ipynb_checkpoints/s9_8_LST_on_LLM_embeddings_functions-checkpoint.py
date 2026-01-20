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




### DEFINE training parameters


from sklearn.model_selection import train_test_split
import warnings
from tqdm import tqdm

autoenc_merged_bool_list = [False, True]
outcome_df=pd.read_csv('../data/tb_1018_20_21_22_30_outcome.csv.gz',index_col=0)
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

## Define string descriptions to add as a prefix for each ds_type, for LLM to know what the variables describe
ds_type_descriptions={'dm':'Demographic descriptors',
                    'mb':'Microbiological test results',
                    'vs':'Vital signs',
                    're':'Chest X-ray findings',
                    'lb':'Laboratory test results',
                    'dr_reg':'Cumulative drug doses taken'}


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

    if X_val is not None and y_val_data_with_index is not None:
        early_stopping=True
    else:
        early_stopping=False        

    
    ### TRAIN MODEL ###
    loss_items_list=[]
    # Define batch size
    batch_size=dense_network_params['batch_size']

    # Create data loaders
    train_dataset=CustomDataset(X_train,y_train_data_with_index)
    train_loader=DataLoader(train_dataset,batch_size=batch_size, shuffle=True)

    if early_stopping==True:
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

        if early_stopping==True:
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

        if early_stopping==True:
            # -------- EARLY STOPPING CHECK --------
            if mean_val_loss < best_val_loss - 1e-6:  # small tolerance
                best_val_loss = mean_val_loss
                best_model_state = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1}. Best val loss: {best_val_loss:.3f}")
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
            
    return model



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
            #self._pca = PCA(n_components=self.pca_var, svd_solver="full", random_state=0)
            self._pca = PCA(n_components=100, svd_solver="full", random_state=0)
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
            #self._pca = PCA(n_components=self.pca_var, svd_solver="full", random_state=0)
            self._pca = PCA(n_components=100,svd_solver="full", random_state=0)
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
    X_train.loc[:,:]=std_scaler.transform(X_train)
    X_test.loc[:,:]=std_scaler.transform(X_test)
    return (X_train, X_test)

##=========================================
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def scale_by_training_data(X_train, X_test):

    scaler = StandardScaler()

    # Fit on training data
    X_train_scaled = scaler.fit_transform(X_train)

    # Transform test data
    X_test_scaled = scaler.transform(X_test)

    # Convert back to DataFrame (with same structure)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

    return X_train_scaled, X_test_scaled



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
        n_of_classes=len(y[outcome_label].unique())
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
    train_roc_auc_scores_all,test_roc_auc_scores_all=[],[]
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
                
        ## Fit model
        if model_name in ['XGBoost','GradientBoost']:
            model.fit(X_train_fold, y_train_fold.values.ravel(),sample_weight=label_weights)
        elif model_name in ['RandomForest','LogisticRegression','SVC','KNN']:
            model.fit(X_train_fold, y_train_fold.values.ravel())
        elif model_name in ['Dense']:
            model = train_dense_network(X_train_fold,
                                        y_train_fold,
                                        updated_dense_network_params,
                                        label_weights_dict,
                                        train_param_comb,
                                        X_val=X_test_fold,
                                        y_val_data_with_index=y_test_fold,
                                        plot_loss_function=True,
                                        verbose=True)
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

                fig,ax=plt.subplots(1,1,figsize=(10,5))
                #print(y_train_fold.values.flatten())
                #print(train_probabilities.flatten())
                sns.scatterplot(y=y_train_fold.values.flatten(),x=train_probabilities.flatten(),ax=ax,s=5)
                #ax.set_title(categorical_map_name,fontsize=10)
                ax.set_ylim(0,None)
                ax.set_ylabel('Loss',fontsize=7)
                fig.suptitle(train_param_comb,fontsize=6,fontweight='bold') 
                

        
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
            train_roc_auc_per_study=y_train_fold.groupby('STUDYID').apply(lambda x:roc_auc_score(x[outcome_label],x['pred']))
            #print(y_test_fold.groupby('STUDYID').apply(lambda x:(x[outcome_label].value_counts())))
            test_roc_auc_per_study=y_test_fold.groupby('STUDYID').apply(lambda x:roc_auc_score(x[outcome_label],x['pred']))

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
        model,label_weights,label_weights_dict=init_model(model_name,X_train,y_train_data,
                                                          k_folds,random_state,outcome_label,
                                                          model_params,
                                                          weight_by_label_freq,
                                                          train_params,
                                                          dense_network_params) 

        model = train_dense_network(X_train,
                                    y_train_data,
                                    dense_network_params,
                                    label_weights_dict,
                                    train_param_comb,
                                    plot_loss_function=False,
                                    verbose=True)
        
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

    
    print('set(idx)&set(de_.index)',set(idx)&set(de_.index))
    
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
    if parameters_for_analysis[data_param_key]['result_cat']=='RESULT_AT_END_OF_TREATMENT': 
        
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
    if parameters_for_analysis[data_param_key]['result_cat']=='RELAPSE'  or 'pred_prob' in parameters_for_analysis[data_param_key]['result_cat']: 
        
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
        ## Set up prediction labels
        id2label={0: "FAVOURABLE", 1: "UNFAVOURABLE"}
        label2id={"FAVOURABLE": 0, "UNFAVOURABLE": 1}
        
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
        pats_with_relapse_df=extract_21_22_relapse_pats()

        ## Set up prediction labels
        #id2label={0: "FAVOURABLE", 1: "UNFAVOURABLE"}
        #label2id={"FAVOURABLE": 0, "UNFAVOURABLE": 1}
    
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


'''
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
    
    return conf_metrics
'''    







#### LSTM FUNCTIONS#########

import copy
import math
from typing import List, Optional, Literal, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    StepLR,
    ReduceLROnPlateau,
)
from torch.nn.functional import sigmoid



import math
import torch
from torch.utils.data import Sampler
import numpy as np


class StratifiedBatchSampler(Sampler):
    """
    Ensures each batch contains a fixed proportion of positive/negative samples.
    For binary classification with imbalance.

    labels: list/array/tensor of 0/1 labels
    batch_size: int
    pos_fraction: fraction of positives per batch (default: match dataset proportion)
    """
    def __init__(self, labels, batch_size, pos_fraction=None, shuffle=True):
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Indices of each class
        self.pos_indices = np.where(self.labels == 1)[0]
        self.neg_indices = np.where(self.labels == 0)[0]

        # Default positive fraction = dataset fraction
        if pos_fraction is None:
            pos_fraction = len(self.pos_indices) / len(self.labels)

        self.pos_per_batch = max(1, int(batch_size * pos_fraction))
        self.neg_per_batch = batch_size - self.pos_per_batch

        # Safety: ensure at least 1 positive per batch
        if self.pos_per_batch == 0:
            self.pos_per_batch = 1
            self.neg_per_batch = batch_size - 1

    def __len__(self):
        return math.ceil(len(self.labels) / self.batch_size)

    def __iter__(self):
        # Shuffle each epoch
        if self.shuffle:
            np.random.shuffle(self.pos_indices)
            np.random.shuffle(self.neg_indices)

        pos_ptr = 0
        neg_ptr = 0

        batches = []

        # Keep creating batches until classes run out
        while pos_ptr < len(self.pos_indices) and neg_ptr < len(self.neg_indices):

            # Collect positives
            pos_batch = self.pos_indices[pos_ptr:pos_ptr + self.pos_per_batch]
            pos_ptr += self.pos_per_batch

            # Collect negatives
            neg_batch = self.neg_indices[neg_ptr:neg_ptr + self.neg_per_batch]
            neg_ptr += self.neg_per_batch

            # If either is empty → stop
            if len(pos_batch) == 0 or len(neg_batch) == 0:
                break

            batch = np.concatenate([pos_batch, neg_batch])

            if self.shuffle:
                np.random.shuffle(batch)

            batches.append(batch)

        return iter(batches)




def return_label_weights(train_labels):
    weight_dict = dict(np.transpose(np.unique(train_labels,return_counts=True)))
    num_neg=float(weight_dict[0])
    num_pos=float(weight_dict[1])
    
    w_pos = (num_neg / num_pos)
    w_neg = 1.0

    return w_neg,w_pos




from torch.utils.data import Dataset, DataLoader
import torch

class TokenDataset(Dataset):
    def __init__(self, token_data, labels):
        """
        token_data: list of dicts with 'input_ids' (1D tensor) and 'attention_mask' (1D tensor)
        labels:     list/array of 0/1
        """
        assert len(token_data) == len(labels)
        self.token_data = token_data
        self.labels = labels

    def __len__(self):
        return len(self.token_data)

    def __getitem__(self, idx):
        td = self.token_data[idx]
        # we keep them on CPU; we'll move to GPU later
        input_ids = td["input_ids"]
        attention_mask = td["attention_mask"]
        label = self.labels[idx]
        return input_ids, attention_mask, label


def collate_tokens(batch):
    """
    batch: list of (input_ids, attention_mask, label)
    Returns:
      input_ids_list:  list of 1D LongTensors (seq_len_i)
      attn_mask_list:  list of 1D LongTensors (seq_len_i)
      labels:          (B,) float tensor
    """
    input_ids_list, attn_list, labels_list = zip(*batch)
    labels = torch.tensor(labels_list, dtype=torch.float32)
    return list(input_ids_list), list(attn_list), labels






class SequenceDataset(Dataset):
    def __init__(self, sequences: List[np.ndarray], labels: List[int]):
        """
        sequences: list of (T_i, H) arrays or tensors
        labels:    list/array of 0/1, len = len(sequences)
        """
        assert len(sequences) == len(labels)
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        if isinstance(seq, np.ndarray):
            seq = torch.from_numpy(seq).float()
        else:
            seq = seq.float()

        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        length = torch.tensor(seq.shape[0], dtype=torch.long)

        return {"seq": seq, "length": length, "label": label}


def collate_fn(batch):
    """
    Pads variable-length sequences in a batch.

    Returns:
      padded_seqs: (B, T_max, H)
      lengths:     (B,)
      labels:      (B,)
    """
    seqs = [item["seq"] for item in batch]
    lengths = torch.tensor([item["length"] for item in batch], dtype=torch.long)
    labels = torch.stack([item["label"] for item in batch])

    padded_seqs = pad_sequence(seqs, batch_first=True)  # (B, T_max, H)

    return padded_seqs, lengths, labels






class TextDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int]):
        assert len(texts) == len(labels)
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

def collate_text(batch):
    texts  = [b[0] for b in batch]
    #labels = torch.tensor([b[1] for b in batch], dtype=torch.float32)
    labels = torch.tensor([int(b[1]) for b in batch], dtype=torch.float32)
    return texts, labels



class BiLSTMAttentionClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.1,
        attention_dim: int = 128,
        fc_layers: list = None,   # e.g. [256, 128] or [128] or []
        fc_dropout: float = 0.2,
        activation: str = "relu",  # "relu" or "gelu"
    ):
        super().__init__()
        self.bidirectional = bidirectional

        # -----------------------------
        #        BiLSTM encoder
        # -----------------------------
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        lstm_out_dim = hidden_dim * (2 if bidirectional else 1)

        # -----------------------------
        #          Attention
        # -----------------------------
        self.attn_W = nn.Linear(lstm_out_dim, attention_dim, bias=True)
        self.attn_v = nn.Linear(attention_dim, 1, bias=False)

        # -----------------------------
        #         MLP head
        # -----------------------------
        if fc_layers is None:
            fc_layers = []   # → single linear layer to output, same as before

        mlp_layers = []
        prev_dim = lstm_out_dim

        # Build hidden FC layers
        for dim in fc_layers:
            mlp_layers.append(nn.Linear(prev_dim, dim))
            if activation == "relu":
                mlp_layers.append(nn.ReLU())
            elif activation == "gelu":
                mlp_layers.append(nn.GELU())
            else:
                raise ValueError("Invalid activation: choose relu or gelu")

            mlp_layers.append(nn.Dropout(fc_dropout))
            prev_dim = dim

        # Final output layer
        mlp_layers.append(nn.Linear(prev_dim, 1))

        self.mlp = nn.Sequential(*mlp_layers)

    # -----------------------------
    #           Forward
    # -----------------------------
    def forward(self, x, lengths):
        """
        x:       (B, T_max, H)
        lengths: (B,)
        """
        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)  # (B, T_max, D)

        B, T_max, D = out.shape
        device = out.device

        # ----- Mask for padded tokens -----
        idxs = torch.arange(T_max, device=device).unsqueeze(0)   # (1, T_max)
        mask = idxs < lengths.unsqueeze(1)                       # (B, T_max)

        # ----- Attention -----
        attn_hidden = torch.tanh(self.attn_W(out))               # (B, T_max, A)
        scores = self.attn_v(attn_hidden).squeeze(-1)            # (B, T_max)

        scores_masked = scores.masked_fill(~mask, float("-inf"))
        attn_weights = F.softmax(scores_masked, dim=1)           # (B, T_max)
        attn_weights_exp = attn_weights.unsqueeze(-1)

        # Weighted sum
        context = torch.sum(attn_weights_exp * out, dim=1)       # (B, D)

        # ----- MLP head -----
        logits = self.mlp(context).squeeze(-1)                   # (B,)

        return logits, attn_weights

class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="none"):
        super().__init__()
        self.alpha = alpha   # float or None
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        targets = targets.view(-1)
        logits  = logits.view(-1)

        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = torch.exp(-bce)

        # α-balancing
        if self.alpha is None:
            alpha_t = 1.0
        else:
            alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)

        focal_term = (1 - p_t) ** self.gamma
        loss = alpha_t * focal_term * bce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def make_scheduler(optimizer, scheduler_name, num_epochs, **kwargs):
    if scheduler_name is None:
        return None

    name = scheduler_name.lower()

    if name == "cosine":
        eta_min = kwargs.get("eta_min", 1e-6)
        return CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=eta_min)

    if name == "step":
        step_size = kwargs.get("step_size", 5)
        gamma     = kwargs.get("gamma", 0.5)
        return StepLR(optimizer, step_size=step_size, gamma=gamma)

    if name == "plateau":
        factor  = kwargs.get("factor", 0.5)
        patience = kwargs.get("patience", 3)
        min_lr  = kwargs.get("min_lr", 1e-6)
        return ReduceLROnPlateau(optimizer, factor=factor, patience=patience, min_lr=min_lr)

    raise ValueError(f"Unknown scheduler: {scheduler_name}")




'''
def train_bilstm_on_embeddings(
    train_embeddings,
    train_labels,
    val_embeddings,
    val_labels,
    input_dim,

    hidden_dim,
    batch_size,
    num_epochs,
    base_lr,
    use_focal_loss,
    focal_alpha=None,
    focal_gamma=2.0,
    device="cuda",
    label_weight_inv_freq: bool = True,
):
    train_ds = SequenceDataset(train_embeddings, train_labels)
    val_ds   = SequenceDataset(val_embeddings,   val_labels)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_seq,
        num_workers=2,      
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_seq,
        num_workers=2,
        pin_memory=True,
    )

    model = BiLSTMAttentionClassifier(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)

    if use_focal_loss:
        loss_fn = BinaryFocalLoss(alpha=focal_alpha, gamma=focal_gamma, reduction="none")
    else:
        loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    best_val_loss = float("inf")
    best_state = None

    if label_weight_inv_freq==True:
        w_neg,w_pos = return_label_weights(train_labels)
    else:
        w_neg,w_pos=1.0,1.0

    from tqdm import trange
    for epoch in trange(1, num_epochs + 1, desc="Epochs"):
        # train
        model.train()
        total_loss = 0.0
        n_samples = 0
        for x, lengths, labels in tqdm(train_loader, desc="Training", leave=False):
            x = x.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits, _ = model(x, lengths)

            loss_per_sample = loss_fn(logits, labels)


            weights = torch.where(labels == 1, w_pos, w_neg)
            
            loss = (loss_per_sample * weights).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            bs = labels.size(0)
            total_loss += loss.item() * bs
            n_samples += bs

        train_loss = total_loss / max(1, n_samples)

        # val
        model.eval()
        total_loss = 0.0
        n_samples = 0
        with torch.no_grad():
            for x, lengths, labels in tqdm(val_loader, desc="Validating", leave=False):
                x = x.to(device)
                lengths = lengths.to(device)
                labels = labels.to(device)
                logits, _ = model(x, lengths)
                loss_per_sample = loss_fn(logits, labels)
                loss = loss_per_sample.mean()
                bs = labels.size(0)
                total_loss += loss.item() * bs
                n_samples += bs

        val_loss = total_loss / max(1, n_samples)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = model.state_dict()
            bad_epochs = 0        # reset counter
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model

'''



def collate_embeddings(seq_emb_list: List[torch.Tensor], labels: torch.Tensor):
    """
    seq_emb_list: list of (T_i, H), maybe in BF16/FP16
    """
    # Convert to float32 for the LSTM
    seq_emb_list = [s.float() for s in seq_emb_list]

    lengths = torch.tensor([s.shape[0] for s in seq_emb_list], dtype=torch.long)
    padded_seqs = pad_sequence(seq_emb_list, batch_first=True)  # (B, T_max, H)

    return padded_seqs, lengths, labels




from tqdm import tqdm

def train_one_epoch_bilstm_with_llm(
    model,
    llm_model,
    tokenizer,
    dataloader,
    device,
    loss_fn,
    optimizer,
    llm_batch_size,
    chunk_len,
    w_pos,
    w_neg, 
    max_grad_norm: Optional[float] = None,
    
):
    model.train()
    total_loss = 0.0
    n_samples  = 0

    pbar = tqdm(dataloader, desc="Training", leave=True)
    
    '''
    Option 1
    for texts, labels in pbar:

        seq_emb_list = []
        for text in texts:
            seq_emb = encode_text_to_hidden_states(
                text=text,
                tokenizer=tokenizer,
                model=llm_model,
                device=device,
                chunk_len=chunk_len,
            )
            seq_emb_list.append(seq_emb)
    '''
    '''
    Option 2
    for texts, labels in pbar:
        seq_emb_list = get_llm_embeddings_for_batch(
            texts=texts,
            tokenizer=tokenizer,
            llm_model=llm_model,
            device=device,
            chunk_len=chunk_len,
            llm_batch_size=llm_batch_size,
        )

        padded_seqs, lengths, labels = collate_embeddings(seq_emb_list, labels)
        padded_seqs = padded_seqs.to(device)
        lengths     = lengths.to(device)
        labels      = labels.to(device)
    '''
    '''
    Option 3
    for input_ids_list, attn_mask_list, labels in pbar:
        seq_emb_list = get_llm_embeddings_for_batch_from_tokens(
            input_ids_list=input_ids_list,
            attn_mask_list=attn_mask_list,
            llm_model=llm_model,
            tokenizer=tokenizer,
            device=device,
            chunk_len=chunk_len,
            llm_batch_size=llm_batch_size,
        )

        padded_seqs, lengths, labels = collate_embeddings(seq_emb_list, labels)
        padded_seqs = padded_seqs.to(device)
        lengths     = lengths.to(device)
        labels      = labels.to(device).float()
    '''



    for texts, labels in pbar:
        start=time.time()
        
        # 1) LLM embeddings for the whole BiLSTM batch (global chunk batching within this batch)
        seq_emb_list = get_llm_embeddings_for_batch(
                texts=texts,
                tokenizer=tokenizer,
                llm_model=llm_model,
                device=device,
                chunk_len=chunk_len,
                llm_batch_size=llm_batch_size,
            )
    
        # 2) Collate into padded batch for BiLSTM
        padded_seqs, lengths, labels = collate_embeddings(seq_emb_list, labels)
        del seq_emb_list
        padded_seqs = padded_seqs.to(device)
        lengths     = lengths.to(device)
        labels      = labels.to(device).float()

        stop=time.time()
        #print('Embedding time:',print_elapsed_time(start,stop))
        
        start=time.time()
        
        optimizer.zero_grad()
        logits, _ = model(padded_seqs, lengths)

        del padded_seqs,lengths

        #loss = loss_fn(logits, labels)

        # Global class weights for this outer split
        weights = torch.where(labels == 1, w_pos, w_neg)

        loss_per_sample = loss_fn(logits, labels)

        # -------------------------------
        # Final weighted loss
        # -------------------------------
        loss = (loss_per_sample * weights).mean()

        del loss_per_sample


        loss.backward()

        if max_grad_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()

        bs = labels.size(0)
        total_loss += loss.item() * bs
        n_samples  += bs

        # Update tqdm bar
        pbar.set_postfix(loss=loss.item())
        #print('loss',loss.item())

        peak_bytes = torch.cuda.max_memory_allocated(device=0)
        #print("Peak PyTorch memory allocated (GB):", peak_bytes / 1024**3)

        # free some memory from seq_emb_list (on CPU mostly, but good practice)
        del labels, logits, loss
        torch.cuda.empty_cache()

        stop=time.time()
        #print('Loss calculation time',print_elapsed_time(start,stop))

    return total_loss / max(n_samples, 1)




@torch.no_grad()
def eval_bilstm_with_llm(
    model,
    llm_model,
    tokenizer,
    dataloader,
    device,
    loss_fn,
    llm_batch_size,
    chunk_len,
):
    model.eval()
    total_loss = 0.0
    n_samples  = 0
    all_probs  = []
    all_labels = []

    pbar = tqdm(dataloader, desc="Validating", leave=True)

    '''
    Option 1
    for texts, labels in pbar:

        seq_emb_list = []
        for text in texts:
            seq_emb = encode_text_to_hidden_states(
                text=text,
                tokenizer=tokenizer,
                model=llm_model,
                device=device,
                chunk_len=chunk_len,
            )
            seq_emb_list.append(seq_emb)
    '''
    '''
    Option 2
    for texts, labels in pbar:
        seq_emb_list = get_llm_embeddings_for_batch(
            texts=texts,
            tokenizer=tokenizer,
            llm_model=llm_model,
            device=device,
            chunk_len=chunk_len,
            llm_batch_size=llm_batch_size,
        )

        padded_seqs, lengths, labels = collate_embeddings(seq_emb_list, labels)
        padded_seqs = padded_seqs.to(device)
        lengths     = lengths.to(device)
        labels      = labels.to(device)
    '''
    '''
    Option 3
    for input_ids_list, attn_mask_list, labels in pbar:
        seq_emb_list = get_llm_embeddings_for_batch_from_tokens(
            input_ids_list=input_ids_list,
            attn_mask_list=attn_mask_list,
            llm_model=llm_model,
            tokenizer=tokenizer,
            device=device,
            chunk_len=chunk_len,
            llm_batch_size=llm_batch_size,
        )

        padded_seqs, lengths, labels = collate_embeddings(seq_emb_list, labels)
        padded_seqs = padded_seqs.to(device)
        lengths     = lengths.to(device)
        labels      = labels.to(device).float()
    '''



    for texts, labels in pbar:

        # 1) LLM embeddings for the whole BiLSTM batch (global chunk batching within this batch)
        seq_emb_list = get_llm_embeddings_for_batch(
                texts=texts,
                tokenizer=tokenizer,
                llm_model=llm_model,
                device=device,
                chunk_len=chunk_len,
                llm_batch_size=llm_batch_size,
            )
    
        # 2) Collate into padded batch for BiLSTM
        padded_seqs, lengths, labels = collate_embeddings(seq_emb_list, labels)
        del seq_emb_list
        padded_seqs = padded_seqs.to(device)
        lengths     = lengths.to(device)
        labels      = labels.to(device).float()
    

        
        logits, _ = model(padded_seqs, lengths)
        #loss = loss_fn(logits, labels)
        del padded_seqs,lengths

        loss_per_sample = loss_fn(logits, labels)
        loss = loss_per_sample.mean()
        del loss_per_sample

        bs = labels.size(0)
        total_loss += loss.item() * bs
        n_samples  += bs

        pbar.set_postfix(loss=loss.item())

        probs = torch.sigmoid(logits).cpu()
        all_probs.append(probs)
        all_labels.append(labels.cpu())

        del  labels, logits, loss
        torch.cuda.empty_cache()

     

    all_probs  = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()
    return total_loss / max(n_samples, 1), all_probs, all_labels









DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def train_full_pipeline(
    train_texts: List[str],
    train_labels: List[int],
    val_texts: List[str],
    val_labels: List[int],
    input_dim: int,
    chunk_len: int,
    llm_batch_size: int,
    hidden_dim: int,
    num_layers: int ,
    batch_size: int,
    num_epochs: int,
    base_lr: float,
    patience: int,
    scheduler_name: str,
    llm_model,
    label_weight_inv_freq: bool,
    device: str,
    use_focal_loss: bool = False,
    focal_alpha: Optional[float] = None,
    focal_gamma: float = 2.0,
    pos_weight: Optional[float] = None,
   
    scheduler_kwargs: Optional[dict] = None,
 
    max_grad_norm: Optional[float] = 5.0,
 
 
):


    '''
    train_tokens = preprocess_texts(train_texts, tokenizer)
    val_tokens   = preprocess_texts(val_texts,   tokenizer)
    
    train_ds = TokenDataset(train_tokens, train_labels)
    val_ds   = TokenDataset(val_tokens,   val_labels)
    
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_tokens)
    
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_tokens)
    
    '''


    sampler = StratifiedBatchSampler(
            labels=train_labels,
            batch_size=batch_size,
            pos_fraction=None,   # use natural class ratio
        )
    train_ds = TextDataset(train_texts, train_labels)
    val_ds   = TextDataset(val_texts,   val_labels)


    train_loader = DataLoader(train_ds, 
                              #batch_size=batch_size, 
                              #shuffle=True,  
                              collate_fn=collate_text,
                              num_workers=2,
                                pin_memory=True,
                              batch_sampler=sampler)
    
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, collate_fn=collate_text,
                             num_workers=2,
                             pin_memory=True,)

    model = BiLSTMAttentionClassifier(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        fc_layers=[256, 128],
        activation="gelu",
        fc_dropout=0.2,
        num_layers=num_layers,
        bidirectional=True,
        dropout=0.1,
        attention_dim=hidden_dim,
    ).to(device)


    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)

    if use_focal_loss:
        loss_fn = BinaryFocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    else:
        if pos_weight is not None:
            pos_w = torch.tensor([pos_weight], dtype=torch.float32, device=device)
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w,reduction='none')
        else:
            loss_fn = nn.BCEWithLogitsLoss(reduction='none')

    scheduler_kwargs = scheduler_kwargs or {}
    scheduler = make_scheduler(optimizer, scheduler_name, num_epochs, **scheduler_kwargs)

 


    # --- Early stopping based on AUC ---
    best_val_auc  = -float("inf")
    best_val_loss = float("inf")
    best_val_ap  = -float("inf")
    best_state    = None
    no_improve    = 0
    #min_delta_auc = 1e-4  # minimum AUC improvement to "count"


    # --- Early stopping based on PR–AUC (average precision) ---
    best_val_ap  = -float("inf")
    best_state   = None
    no_improve   = 0
    min_delta_ap = 1e-4  # minimum AP improvement to "count"


    from tqdm import trange
    #from tqdm import tqdm

    if label_weight_inv_freq==True:
        w_neg,w_pos = return_label_weights(train_labels)
    else:
        w_neg,w_pos=1.0,1.0

    history = []

    for epoch in trange(1, num_epochs + 1, desc="Epochs"):
    #for epoch in tqdm(1, num_epochs + 1, desc="Epochs"):

        train_loss = train_one_epoch_bilstm_with_llm(
            model=model,
            llm_model=llm_model,
            tokenizer=tokenizer,
            dataloader=train_loader,
            device=device,
            loss_fn=loss_fn,
            optimizer=optimizer,
            chunk_len=chunk_len,
            max_grad_norm=max_grad_norm,
            w_neg=w_neg,
            w_pos=w_pos,
            llm_batch_size=llm_batch_size,
            
        )

        val_loss, val_probs, val_labels = eval_bilstm_with_llm(
            model=model,
            llm_model=llm_model,
            tokenizer=tokenizer,
            dataloader=val_loader,
            device=device,
            loss_fn=loss_fn,
            chunk_len=chunk_len,
            llm_batch_size=llm_batch_size,
        )

        try:
            val_ap = average_precision_score(val_labels, val_probs)
        except ValueError:
            val_ap = np.nan

         # --- Step scheduler ---
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                # IMPORTANT: configure ReduceLROnPlateau with mode="max"
                scheduler.step(val_ap)
            else:
                scheduler.step()

        lr = optimizer.param_groups[0]["lr"]

         # --- Compute ROC–AUC and PR–AUC on validation set ---
        try:
            val_auc_roc = roc_auc_score(val_labels, val_probs)
        except ValueError:
            val_auc_roc = np.nan




        
        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"ROC-AUC={val_auc_roc:.4f} | "
            f"PR-AUC={val_ap:.4f} | "
            f"lr={lr:.2e}"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss":  val_loss,
            "val_auc_roc": val_auc_roc,
            "val_ap":   val_ap,
            "lr": lr,
            "val_probs":  val_probs,
            "val_labels": val_labels,
        })

        # --- Early stopping & best model selection based on PR–AUC ---
        if not np.isnan(val_ap) and val_ap > best_val_ap + min_delta_ap:
            best_val_ap = val_ap
            best_state  = copy.deepcopy(model.state_dict())
            no_improve  = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch} (no PR–AUC improvement)")
                break
    
    torch.cuda.empty_cache()
    
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history





@torch.no_grad()
def run_model_on_test_set(
    model,
    llm_model,
    tokenizer,
    test_texts,
    test_labels,
    device,
    chunk_len,
    llm_batch_size,
    batch_size,
):
    from tqdm import tqdm
    """
    model: trained BiLSTM model
    llm_model: AutoModel (base LLM)
    tokenizer: HuggingFace tokenizer
    test_texts: list[str]
    test_labels: list[int or float]
    device: "cuda" or "cpu"
    """

    model.eval()
    llm_model.eval()

    pred_probs_list = []
    true_list = []

    # Create dataset-like batching
    def batch_iter(data, labels, batch_size):
        for i in range(0, len(data), batch_size):
            yield data[i:i+batch_size], labels[i:i+batch_size]

    for texts_batch, labels_batch in tqdm(
        batch_iter(test_texts, test_labels, batch_size),
        desc="Running test inference",
    ):
        # (1) Compute LLM embeddings (list of length B; tensors of shape (T_i, H))
        seq_emb_list = get_llm_embeddings_for_batch(
            texts=texts_batch,
            tokenizer=tokenizer,
            llm_model=llm_model,
            device=device,
            chunk_len=chunk_len,
            llm_batch_size=llm_batch_size,
        )

        # (2) Collate into padded batch for BiLSTM
        padded_seqs, lengths, labels_tensor = collate_embeddings(
            seq_emb_list,
            torch.tensor(labels_batch, dtype=torch.float32),
        )
        del seq_emb_list
        
        padded_seqs = padded_seqs.to(device)
        lengths     = lengths.to(device)

        # (3) Forward through BiLSTM classifier
        logits, _ = model(padded_seqs, lengths)     # (B,)
        probs  = sigmoid(logits).cpu().numpy()      # convert to probabilities

        del padded_seqs,logits, lengths
        
        # (4) Store outputs
        pred_probs_list.append(probs)
        true_list.append(labels_tensor.numpy())

        del probs

    # Concatenate output batches
    import numpy as np
    pred_probs = np.concatenate(pred_probs_list)
    true_labels = np.concatenate(true_list)

    return pred_probs, true_labels





###### LLM FUNCTIONS #######


from torch.nn.utils.rnn import pad_sequence
from torch.cuda.amp import autocast
import torch
from tqdm.auto import tqdm




from tqdm import tqdm

def preprocess_texts(texts, tokenizer):
    from tqdm.auto import tqdm
    """
    Tokenize all texts once. Returns a list of dicts with input_ids and attention_mask on CPU.
    """
    token_data = []
    for t in tqdm(texts, desc="Tokenizing texts"):
        enc = tokenizer(
            t,
            return_tensors="pt",
            truncation=False,
            padding=False,
        )
        # store as 1D tensors (seq_len,)
        token_data.append({
            "input_ids": enc["input_ids"].squeeze(0).clone(),         # LongTensor
            "attention_mask": enc["attention_mask"].squeeze(0).clone()
        })
    return token_data








from torch.nn.utils.rnn import pad_sequence
from torch.cuda.amp import autocast
import torch

def get_llm_embeddings_for_batch(
    texts,
    tokenizer,
    llm_model,
    device,
    chunk_len: int,
    llm_batch_size: int,
):
    """
    texts: list[str] of length B_lstm
    Returns: list of length B_lstm; each element is a tensor (T_i, H) on CPU, float32.
    """

    # 1) Tokenize and chunk each patient
    all_patients_chunks = []  # list[list[(ids_chunk, mask_chunk)]]
    for text in texts:
        enc = tokenizer(
            text,
            return_tensors="pt",
            truncation=False,
            padding=False,
        )
        ids  = enc["input_ids"]      # (1, T)
        mask = enc["attention_mask"] # (1, T)

        chunks = []
        for i in range(0, ids.shape[1], chunk_len):
            chunks.append((
                ids[:, i:i+chunk_len],
                mask[:, i:i+chunk_len],
            ))
        all_patients_chunks.append(chunks)

    # 2) Flatten into a list of jobs: (patient_idx, ids_chunk, mask_chunk)
    jobs = []
    for pid, chunks in enumerate(all_patients_chunks):
        for (ids_chunk, mask_chunk) in chunks:
            jobs.append((pid, ids_chunk, mask_chunk))

    # 3) Prepare storage per patient
    patient_emb_lists = [[] for _ in range(len(texts))]

    llm_model.eval()
    with torch.no_grad():
        for i in range(0, len(jobs), llm_batch_size):
            batch_jobs = jobs[i:i+llm_batch_size]

            # Take 1D token sequences for each chunk
            ids_1d_list  = [j[1].squeeze(0) for j in batch_jobs]  # (seq_len,)
            mask_1d_list = [j[2].squeeze(0) for j in batch_jobs]

            # Pad chunks within this job batch
            ids_padded = pad_sequence(
                ids_1d_list,
                batch_first=True,
                padding_value=tokenizer.pad_token_id,
            )
            mask_padded = pad_sequence(
                mask_1d_list,
                batch_first=True,
                padding_value=0,
            )

            ids_batch  = ids_padded.to(device)
            mask_batch = mask_padded.to(device)

            # LLM forward (mixed precision)
            with autocast(dtype=torch.bfloat16):
                out = llm_model(
                    input_ids=ids_batch,
                    attention_mask=mask_batch,
                    return_dict=True,
                )
                last_hidden = out.last_hidden_state  # (B_chunk, T_max, H)

            # Move to CPU float32
            last_hidden = last_hidden.float().cpu()

            # Trim padding and assign to patients
            for (job, ids_1d, emb) in zip(batch_jobs, ids_1d_list, last_hidden):
                pid = job[0]
                orig_len = ids_1d.shape[0]
                emb_trimmed = emb[:orig_len]  # (orig_len, H)
                patient_emb_lists[pid].append(emb_trimmed)

            del out, last_hidden, ids_batch, mask_batch

    # 4) Concatenate chunks to full sequence per patient
    seq_emb_list = [torch.cat(chunks, dim=0) for chunks in patient_emb_lists]
    return seq_emb_list










from torch.nn.utils.rnn import pad_sequence
from torch.cuda.amp import autocast

def get_llm_embeddings_for_batch_from_tokens(
    input_ids_list,
    attn_mask_list,
    llm_model,
    tokenizer,
    device,
    chunk_len: int,
    llm_batch_size: int,
):
    """
    input_ids_list: list of 1D LongTensors, each (T_i,)
    attn_mask_list: list of 1D LongTensors, each (T_i,)
    Returns: list of tensors [ (T_i, H), ... ] on CPU, float32
    """

    # 1) Chunk each patient
    all_patients_chunks = []  # list of list[(ids_chunk, mask_chunk)]
    for ids_1d, mask_1d in zip(input_ids_list, attn_mask_list):
        ids_1d = ids_1d.view(1, -1)   # (1, T)
        mask_1d = mask_1d.view(1, -1)

        chunks = []
        for i in range(0, ids_1d.shape[1], chunk_len):
            chunks.append((
                ids_1d[:, i:i+chunk_len],
                mask_1d[:, i:i+chunk_len],
            ))
        all_patients_chunks.append(chunks)

    # 2) Flatten into jobs: (pid, ids_chunk, mask_chunk)
    jobs = []
    for pid, chunks in enumerate(all_patients_chunks):
        for (ids, mask) in chunks:
            jobs.append((pid, ids, mask))

    patient_emb_lists = [[] for _ in range(len(input_ids_list))]

    llm_model.eval()
    with torch.no_grad():
        for i in range(0, len(jobs), llm_batch_size):
            batch_jobs = jobs[i:i+llm_batch_size]

            # Extract 1D tensors for each chunk
            ids_1d_list  = [j[1].squeeze(0) for j in batch_jobs]  # (seq_len_i,)
            mask_1d_list = [j[2].squeeze(0) for j in batch_jobs]

            # Pad to max length in this chunk batch
            ids_padded = pad_sequence(
                ids_1d_list,
                batch_first=True,
                padding_value=tokenizer.pad_token_id,
            )
            mask_padded = pad_sequence(
                mask_1d_list,
                batch_first=True,
                padding_value=0,
            )

            ids_batch  = ids_padded.to(device)
            mask_batch = mask_padded.to(device)

            # LLM forward (mixed precision)
            with autocast(dtype=torch.bfloat16):           

                outputs = llm_model(
                        input_ids=ids_batch,
                        attention_mask=mask_batch,
                        #output_hidden_states=True,
                        output_hidden_states=False,                    
                        return_dict=True,
                    )
                #last_hidden = out.hidden_states[-1]  # (B, T_max_in_batch, H)
                last_hidden = outputs.last_hidden_state 

            last_hidden = last_hidden.float().cpu()

            # Trim padding and assign to correct patient
            for (job, ids_1d, emb) in zip(batch_jobs, ids_1d_list, last_hidden):
                pid = job[0]
                orig_len = ids_1d.shape[0]
                emb_trimmed = emb[:orig_len]  # (orig_len, H)
                patient_emb_lists[pid].append(emb_trimmed)

            del outputs, last_hidden, ids_batch, mask_batch
            #torch.cuda.empty_cache()

    # 3) Concatenate chunks per patient
    seq_emb_list = [torch.cat(chunks, dim=0) for chunks in patient_emb_lists]
    return seq_emb_list














from torch.cuda.amp import autocast
from torch.nn.utils.rnn import pad_sequence
from torch.cuda.amp import autocast

def get_llm_embeddings_for_batch(
    texts,
    tokenizer,
    llm_model,
    device,
    chunk_len: int,
    llm_batch_size: int,
):
    # 1) Chunk texts
    all_patients_chunks = []
    for text in texts:
        enc = tokenizer(text, return_tensors="pt", truncation=False, padding=False)
        ids  = enc["input_ids"]
        mask = enc["attention_mask"]

        chunks = []
        for i in range(0, ids.shape[1], chunk_len):
            chunks.append((
                ids[:, i:i+chunk_len], 
                mask[:, i:i+chunk_len]
            ))
        all_patients_chunks.append(chunks)

    # 2) Flatten into jobs: (pid, ids, mask)
    jobs = []
    for pid, chunks in enumerate(all_patients_chunks):
        for ids, mask in chunks:
            jobs.append((pid, ids, mask))

    # Where to store resulting embeddings
    patient_emb_lists = [[] for _ in range(len(texts))]

    llm_model.eval()
    with torch.no_grad():
        for i in range(0, len(jobs), llm_batch_size):
            batch_jobs = jobs[i:i+llm_batch_size]

            # ---- NEW: PAD SEQUENCES BEFORE LLM ----
            ids_list  = [j[1].squeeze(0) for j in batch_jobs]   # shape (seq_len,)
            masks_list = [j[2].squeeze(0) for j in batch_jobs]

            # pad to max seq_len in this batch
            ids_padded  = pad_sequence(ids_list,  batch_first=True, padding_value=tokenizer.pad_token_id)
            mask_padded = pad_sequence(masks_list, batch_first=True, padding_value=0)

            # restore batch dims
            ids_batch  = ids_padded.to(device)
            mask_batch = mask_padded.to(device)

            # LLM forward
            with autocast(dtype=torch.bfloat16):
                out = llm_model(
                    input_ids=ids_batch,
                    attention_mask=mask_batch,
                    #output_hidden_states=True,
                    output_hidden_states=False,                    
                    return_dict=True,
                )
                #last_hidden = out.hidden_states[-1]  # (B, T_max_in_batch, H)
                last_hidden = out.last_hidden_state 

            # move to CPU float32
            last_hidden = last_hidden.float().cpu()

            # ---- TRIM per sample back to correct original chunk lengths ----
            for (job, ids, emb) in zip(batch_jobs, ids_list, last_hidden):
                pid = job[0]
                orig_len = ids.shape[0]
                emb_trimmed = emb[:orig_len]  # remove padding
                patient_emb_lists[pid].append(emb_trimmed)

            del out, last_hidden, ids_batch, mask_batch
            #torch.cuda.empty_cache()

    # merge chunk embeddings per patient
    seq_emb_list = [torch.cat(chunks, dim=0) for chunks in patient_emb_lists]
    return seq_emb_list
















def chunk_ids_and_mask(input_ids, attention_mask, max_len: int = 512):
    """
    input_ids, attention_mask: shape (1, T)
    Returns lists of chunked ids/masks.
    """
    chunks_ids = []
    chunks_mask = []

    total_len = input_ids.shape[1]

    for i in range(0, total_len, max_len):
        chunks_ids.append(input_ids[:, i:i+max_len])
        chunks_mask.append(attention_mask[:, i:i+max_len])

    return chunks_ids, chunks_mask

@torch.no_grad()
def encode_text_to_hidden_states(
    text: str,
    tokenizer,
    model,
    device: str = DEVICE,
    chunk_len: int = 512,
):
    """
    Returns a tensor of shape (T_total, H) with last hidden layer for all tokens.
    """
    # Tokenize once, no truncation
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=False,
        padding=False,
    )
    input_ids = encoded["input_ids"].to(device)      # (1, T)
    attention_mask = encoded["attention_mask"].to(device)

    # Chunk into windows along sequence dimension
    chunks_ids, chunks_mask = chunk_ids_and_mask(input_ids, attention_mask, max_len=chunk_len)

    all_hidden = []

    for ids, mask in zip(chunks_ids, chunks_mask):
        outputs = model(
            input_ids=ids,
            attention_mask=mask,
            output_hidden_states=True,
            return_dict=True,
        )
        # last hidden layer: (1, chunk_T, H) -> (chunk_T, H)
        last_hidden = outputs.hidden_states[-1].squeeze(0)
        all_hidden.append(last_hidden.cpu())  # move to CPU to save VRAM

        del outputs
        #torch.cuda.empty_cache()

    # Concatenate chunks: (T_total, H)
    full_seq = torch.cat(all_hidden, dim=0)
    return full_seq













### -==============================================

def init_tokenizer():
    
    llm_model_names=['BioMistral/BioMistral-7B',"epfl-llm/meditron-7b",'FremyCompany/BioLORD-2023',
                     "google/long-t5-local-base",'google/long-t5-tglobal-large',"epfl-llm/meditron-70b"]
    fine_tuned_tags=['base','fine_tuned']
    
    for llm_model_name in llm_model_names[:1]:
        print(llm_model_name)
        for fine_tuned_tag in fine_tuned_tags[:1]:
             print(fine_tuned_tag)
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

    return tokenizer


tokenizer = init_tokenizer()

### -==============================================
def load_input_texts_and_labels(ds_types_merged,dataset_param_key,target_df,prompt,period_end_day,data_inclusion_type,outcome_label):
    if data_inclusion_type=='baseline_vars_ext':
        data_inclusion_type='all_days'

    key='tb21_22_2984_pats_22_vars_result_at_end_of_treatment'
    if ds_types_merged==True:
        
        #fn=f'../data/{dataset_param_key}_input_dict_all_ds_types_merged.json'
        
        fn=f'../data/{key}_{period_end_day}_days_input_dict_all_ds_types_merged.json'
        fn=f'../data/{key}_{period_end_day}_days_{data_inclusion_type}_input_dict_all_ds_types_merged.json'
        input_dict=json.load(open(fn))

        input_dict_filt = {key: input_dict[key] for key in target_df.index}
        
        all_labels=target_df.loc[[*input_dict_filt],outcome_label].values.tolist() #.astype(int)
        all_texts=np.array(list(input_dict_filt.values())).tolist()
        all_texts=[f'{prompt} {text}' for text in all_texts]
        
        pat_ids=list(input_dict_filt.keys())
        
        return pat_ids,None,all_texts,all_labels
        
    if ds_types_merged==False: 
        #fn=f'../data/{dataset_param_key}_input_dict_ds_types_seperate.json'
        fn=f'../data/{key}_{period_end_day}_days_input_dict_ds_types_seperate.json'
        fn=f'../data/{key}_{period_end_day}_days_{data_inclusion_type}_input_dict_ds_types_seperate.json'
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


    


### -==============================================
## DEFINE TIMEPOINT OF PREDICTION AND ADD IT IN A PROMPT TO THE BEGINNING OF EACH INPUT
def create_prompt(data_inclusion_type):
    #timepoint=outcome_label.replace('_',' ').lower().split(' at')[-1]
    #if timepoint==' end of treatment':
        #timepoint='6 months'

    #if data_inclusion_type=='baseline_last_day':
    #prompt=f'[INST] The following data originates from a patient with pulmonary tuberculosis, participating in a Phase 3 clinical trial. Please predict the outcome of the therapy {timepoint} after therapy induction as FAVOURABLE or UNFAVOURABLE. [/INST]'
    #prompt=f'[INST] The following data originates from a patient with pulmonary tuberculosis, participating in a Phase 3 clinical trial. Please summarise the condition of the patient. [/INST]'
    prompt=''
    return prompt




def load_llm():
    
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
    
    
    from transformers import AutoModel,AutoModelForSequenceClassification,LongT5EncoderModel, AutoConfig
    from peft import LoraConfig, TaskType,get_peft_model,PeftModel
    
    # Define the path where your model is saved
    cache_dir='../huggingface_cache'
    
    
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
                '''
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
                '''                                                                         

                model=AutoModel.from_pretrained(llm_model_name,
                                                 cache_dir=cache_dir,
                                                 #output_hidden_states=True,
                                                 #output_attentions=True,
                                                 torch_dtype=torch.bfloat16,#orch.float16,                                                        
                                                 num_labels=len(id2label.keys()),
                                                 id2label=id2label, 
                                                 label2id=label2id,
                                                 device_map='auto',
                                                 pad_token_id=2,
                                                 force_download=False, 
                                                 resume_download=False)

            
            
            print(f'{llm_model_name} base model loaded')
        
            ## IF INFERENCE WISHED BY FINE-TUNED MODEL, ADD & MERGE FINE-TUNED PEFT MODEL ADAPTER
            #peft_model_path=f"../data/{llm_model_name}_{fine_tuned_tag}_num_of_epochs_{num_of_epochs}_{period_end_day}_days"
            peft_model_path=f"../data/{llm_model_name}_{fine_tuned_tag}_num_of_epochs_{num_of_epochs}"#_{period_end_day}_days"
            
            if fine_tuned_tag=='fine_tuned':
                model=PeftModel.from_pretrained(model,peft_model_path)
                model=model.merge_and_unload()
                
                print(f'{llm_model_name}_{fine_tuned_tag} PEFT fine-tuned adapter added')
                
    return model









