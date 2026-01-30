import pandas as pd
import warnings
# Ignoring FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)
#warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)
import numpy as np
import os
import time
import pickle
from matplotlib import pyplot as plt
import seaborn as sns
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score,f1_score,mean_squared_error,roc_auc_score
from sklearn.model_selection import GridSearchCV,KFold,train_test_split

from itertools import chain
import math

from sklearn.preprocessing import StandardScaler,MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader,Dataset
from torch.nn.utils.rnn import pad_sequence,pack_padded_sequence,pad_packed_sequence
from torch.optim.lr_scheduler import CosineAnnealingLR, CyclicLR
from torch.optim.swa_utils import AveragedModel
import copy
#from pytorch_model_summary import summary


## LOAD DATAFRAME CONTATINING TARGET OUTCOMES
outcome_df=pd.read_csv('../data/tb_1018_20_21_22_30_outcome.csv.gz',index_col=0)
outcome_df=outcome_df.set_index('USUBJID',drop=True)
outcome_df=outcome_df.rename(columns={'UNFAVOURABLE_OUTCOME_CATEGORY_AT_18_MONTHS':'UNFAVOUR_CAT_AT_18_MONTHS'})


parameters_for_analysis={'tb21_22_2984_pats_22_vars_result_at_end_of_treatment':{
                            'fn':'tb21_22_2984_pats_22_vars_result_at_end_of_treatment',
                              'training_days':120,
                            'result_cat':'RESULT_AT_END_OF_TREATMENT'},
            
                        'tb21_22_2984_pats_22_vars_relapse':{
                            'fn':'tb21_22_2984_pats_22_vars_result_at_end_of_treatment',
                              'training_days':120,
                            'result_cat':'RELAPSE'}

                         }


# 1. DEFINE FUNCTIONS FOR LSTM

'''
DEPR: too slow
# Define a custom Dataset for handling grouped training data
class Grouped_Train_Dataset(Dataset):
    def __init__(self, X,y,model_complex,columns_to_drop):
        self.X = X
        self.y = y
        self.training_days = training_days
        self.patients=X['USUBJID'].unique()
        self.model_complex=model_complex
        #print('Training data shape',self.X.shape)

    def __len__(self):
        return len(self.X['COMBNUM'].unique())

    def __getitem__(self, idx):
        combnum_dict=dict(zip(np.arange(len(self.X['COMBNUM'].unique())),self.X['COMBNUM'].unique()))
        combnum=combnum_dict[idx]
        X_patient_data=self.X[self.X['COMBNUM']==combnum]
        pat_id=X_patient_data['USUBJID'].unique()
        y_patient_data=self.y[self.y.index.get_level_values('USUBJID').isin(pat_id)]

        if self.model_complex=='day_only':
            ### DROP COLUMNS FOR BASE MODEL => STUDY|ARM|USUBJID will be dropped later in the script
            X_patient_data=X_patient_data.loc[:,X_patient_data.columns.str.contains('DAY|COMBNUM|STUDY|ARM|USUBJID',regex=True)] 

        if self.model_complex=='regimen_only':       
            ### DROP COLUMNS FOR BASE MODEL => STUDY|ARM|USUBJID|DAY will be dropped later in the script
            X_patient_data=X_patient_data.loc[:,X_patient_data.columns.str.contains('dr_reg|COMBNUM|STUDY|ARM|USUBJID',regex=True)]      
        
        X_patient_data=X_patient_data.loc[:,~X_patient_data.columns.str.contains('|'.join(columns_to_drop +['COMBNUM','USUBJID']),regex=True)]      
        #print('X_patient_data.columns',X_patient_data.columns)
        
        X_patient_data=torch.tensor(X_patient_data.values, dtype=torch.float32)
        y_patient_data=torch.tensor(y_patient_data.values, dtype=torch.float32)


        #print('X_patient_data',X_patient_data.shape)
        #print('y_patient_data',y_patient_data.shape)

        return X_patient_data, y_patient_data    


    def collate_fn(self, batch):
        inputs, targets= zip(*batch)
        inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=0)
        targets = torch.stack(targets)
        seq_lens=torch.tensor([len(seq) for seq in inputs],dtype=torch.float32)
        return inputs_padded, targets, seq_lens

'''


## Drop patients who have their last data at an earlier timepoint than threshold
therapy_day_thr=80

period_end_days=['baseline',31,62,93,125,160,'all']

## Set up prediction labels
id2label={0: "FAVOURABLE", 1: "UNFAVOURABLE"}
label2id={"FAVOURABLE": 0, "UNFAVOURABLE": 1}


import time

outcome_df=pd.read_csv('../data/tb_1018_20_21_22_30_outcome.csv.gz',index_col=0)
outcome_df=outcome_df.set_index('USUBJID',drop=True)
outcome_df=outcome_df.rename(columns={'UNFAVOURABLE_OUTCOME_CATEGORY_AT_18_MONTHS':'UNFAVOUR_CAT_AT_18_MONTHS'})

## List of temporal dataframe names to drop from loaded dataframe (as they are highly sparse, perhaps only add noise?)
temporal_data_names=['ae_','cmind_','cmdos_','cmday_','mh']

#columns_to_drop=['USUBJID','ARM','STUDYID','DAY','index']
columns_to_drop=['ARM','STUDYID','DAY','index'][:-1]
temp_cols_to_drop=['ae','ce','mh','cm']


##========
## More speed efficient
class Grouped_Train_Dataset(Dataset):
    def __init__(self, X, y, model_complex, columns_to_drop):
        self.model_complex = model_complex

        # Precompute unique COMBNUMs and a mapping for fast access
        self.combnums = X['COMBNUM'].unique()
        self.combn_to_data = {}  # cache to avoid recomputation

        # Pre-filter relevant columns ONCE depending on model complexity
        if model_complex == 'day_only':
            include_cols = X.columns[X.columns.str.contains('DAY|COMBNUM|STUDY|ARM|USUBJID')]
        elif model_complex == 'regimen_only':
            include_cols = X.columns[X.columns.str.contains('dr_reg|COMBNUM|STUDY|ARM|USUBJID')]
        else:
            include_cols = X.columns

        # Drop columns
        exclude_cols = columns_to_drop + ['COMBNUM', 'USUBJID']
        final_cols = [col for col in include_cols if not any(ex in col for ex in exclude_cols)]

        self.X = X
        self.y = y
        self.final_cols = final_cols

        # Group data by COMBNUM in advance for fast retrieval
        grouped = X.groupby('COMBNUM')
        for comb in self.combnums:
            df = grouped.get_group(comb)
            df_selected = df[final_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
            self.combn_to_data[comb] = torch.tensor(df_selected.values, dtype=torch.float32)

        # Index y data by patient ID for faster slicing
        #print('y within init:')
        #print(y,'\n')
        self.y_by_patient = {
            pat: torch.tensor(val.values, dtype=torch.float32)
            for pat, val in y.groupby(level='USUBJID')
        }

        self.combn_to_pat = {}
        for comb in self.combnums:
            usubjids = X[X['COMBNUM'] == comb]['USUBJID'].unique()
            if len(usubjids) != 1:
                raise ValueError(f"COMBNUM {comb} maps to multiple USUBJIDs: {usubjids}")
            self.combn_to_pat[comb] = usubjids[0]

        del X
        del y
        self.X = None
        self.y = None

    def __len__(self):
        return len(self.combnums)

    def __getitem__(self, idx):
        comb = self.combnums[idx]
        X_tensor = self.combn_to_data[comb]
        pat_id = self.combn_to_pat[comb]
        y_tensor = self.y_by_patient[pat_id]
        return X_tensor, y_tensor

    def collate_fn(self, batch):
        inputs, targets = zip(*batch)
        inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=0)
        targets = torch.stack(targets)
        #seq_lens = torch.tensor([len(seq) for seq in inputs], dtype=torch.float32)
        seq_lens = torch.tensor([len(seq) for seq in inputs], dtype=torch.long) 
        return inputs_padded, targets, seq_lens



##### =======================================
# Define a custom Dataset for handling grouped test data for the training method training_on__whole_data
class Grouped_Test_Dataset_whole_data(Dataset):
    def __init__(self, X_test,y_test,model_complex,columns_to_drop,dataset_type):
        self.X_test=X_test
        self.y_test=y_test
        self.model_complex=model_complex
        #self.training_days = training_days
        #self.std_training_days=std_training_days.flatten()[0]      
        
        self.X_test=self.X_test.reset_index(drop=True)     

    def __len__(self):
        return len(self.X_test['COMBNUM'].unique())

    def __getitem__(self, idx):
        #patient_id=self.X_test.loc[idx,'USUBJID']
        combnum_dict=dict(zip(np.arange(len(self.X_test['COMBNUM'].unique())),self.X_test['COMBNUM'].unique()))
        combnum=combnum_dict[idx]
        
        X_testing_data=self.X_test[self.X_test['COMBNUM']==combnum]
        pat_id=X_testing_data['USUBJID'].unique()
        
    
        if self.model_complex=='day_only':
            ### DROP COLUMNS FOR BASE MODEL => STUDY|ARM|USUBJID will be dropped later in the script
            X_testing_data=X_testing_data.loc[:,X_testing_data.columns.str.contains('DAY|COMBNUM|STUDY|ARM|USUBJID|mb_Time to Detection_STD_NUM_RESULT',regex=True)] 

        if self.model_complex=='regimen_only':       
            ### DROP COLUMNS FOR BASE MODEL => STUDY|ARM|USUBJID|DAY will be dropped later in the script
            X_testing_data=X_testing_data.loc[:,X_testing_data.columns.str.contains('dr_reg|COMBNUM|STUDY|ARM|USUBJID|mb_Time to Detection_STD_NUM_RESULT',regex=True)]                                        

        ## Drop predefined categoircal columns
        X_testing_data=X_testing_data.loc[:,~X_testing_data.columns.str.contains('|'.join(columns_to_drop +['COMBNUM','USUBJID']),regex=True)]  
        
        # Subset the y data for the specific patient
        #y_testing_data=self.y_test[(self.y_test.index.get_level_values('USUBJID')==patient_id)]#.drop(columns=['COMBNUM']) 
        y_testing_data=self.y_test[self.y_test.index.get_level_values('USUBJID').isin(pat_id)]
                                  
        X_testing_data=torch.tensor(X_testing_data.values, dtype=torch.float32)
        y_testing_data=torch.tensor(y_testing_data.values, dtype=torch.float32)

        return X_testing_data, y_testing_data  
        
    def collate_fn(self, batch):
        inputs, targets= zip(*batch)
        inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=0)
        targets = torch.stack(targets)
        seq_lens=torch.tensor([len(seq) for seq in inputs],dtype=torch.float32)
        return inputs_padded, targets

##### =======================================
class Grouped_Test_Dataset_whole_data(Dataset):
    def __init__(self, X_test, y_test, model_complex, columns_to_drop, dataset_type):
        self.model_complex = model_complex
        self.dataset_type = dataset_type
        #self.training_days = training_days

        self.combn_list = X_test['COMBNUM'].unique()

        # Column filtering based on model complexity
        if model_complex == 'day_only':
            include_cols = X_test.columns[X_test.columns.str.contains(
                'DAY|COMBNUM|STUDY|ARM|USUBJID|mb_Time to Detection_STD_NUM_RESULT')]
        elif model_complex == 'regimen_only':
            include_cols = X_test.columns[X_test.columns.str.contains(
                'dr_reg|COMBNUM|STUDY|ARM|USUBJID|mb_Time to Detection_STD_NUM_RESULT')]
        else:
            include_cols = X_test.columns

        # Drop unwanted columns
        exclude_cols = columns_to_drop + ['COMBNUM', 'USUBJID']
        final_cols = [col for col in include_cols if not any(ex in col for ex in exclude_cols)]
        self.final_cols = final_cols

        # Group and precompute tensors
        self.data = []
        grouped_X = X_test.groupby('COMBNUM')
        grouped_y = y_test.groupby(level='USUBJID')

        for comb in self.combn_list:
            x_df = grouped_X.get_group(comb)
            usubjids = x_df['USUBJID'].unique()

            if len(usubjids) != 1:
                raise ValueError(f"❌ COMBNUM {comb} maps to multiple USUBJIDs: {usubjids}")

            pat_id = usubjids[0]

            # Clean X
            x_tensor = torch.tensor(
                x_df[final_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values,
                dtype=torch.float32
            )

            # Clean y
            if pat_id in grouped_y.groups:
                y_tensor = torch.tensor(grouped_y.get_group(pat_id).values, dtype=torch.float32)
            else:
                raise ValueError(f"❌ No matching label found for USUBJID: {pat_id}")

            self.data.append((x_tensor, y_tensor))

        # Clean up memory
        del X_test
        del y_test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def collate_fn(self, batch):
        inputs, targets = zip(*batch)
        inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=0)
        targets = torch.stack(targets)
        return inputs_padded, targets




##### =======================================
### LSTM for forecasting 
class LSTM_forecast(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size,fc_hidden_dims,device,dropout_prob,bidirectional):
        super().__init__()
        ## Init LSTM model
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout_prob,batch_first=True,
                            bidirectional=bidirectional)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_prob)

        ## Init fully connected (FC) layers
        dims=[hidden_size] + fc_hidden_dims + [output_size]
        layers=[]
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.BatchNorm1d(num_features=dims[i+1]))
            layers.append(nn.ReLU())            
            #layers.append(nn.Dropout(dropout_prob))
        
        # Drop the BatchNorm + Dropout layer before th Softmax
        layers=layers[:-1]
        # Replace the last activation function
        layers[-1]=nn.Softmax(dim=1)

        self.fc=nn.Sequential(*layers)
        '''
        self.fc = nn.Sequential(nn.Linear(hidden_size, int(hidden_size/2)),
                                nn.ReLU(),
                                nn.Dropout(dropout_prob), 
                                nn.Linear(int(hidden_size/2),int(hidden_size/4)),
                                nn.ReLU(),
                                nn.Dropout(dropout_prob), 
                                nn.Linear(int(hidden_size/4),output_size),
                                nn.Softmax(dim=2))
        '''                                
                                   
        self.device=device
        
    def forward(self, x, seq_lens,mode):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        
        if mode=='training':
            lstm_input=pack_padded_sequence(x,seq_lens,enforce_sorted=False,batch_first=True)
        if mode=='testing':
            lstm_input=x #x.view(1,1,x.shape[0])

        output,(ht,ct) = self.lstm(lstm_input)
        ## Extract the last output of the LSTM layers (as there can be multiple)
        fc_input=ht[-1].unsqueeze(0)
        
        ## Permutate, to get the batch size as first dimension so reshape from (1,batch_size,hidden_size)
        #  into (batch_size,1,hidden_size)
        #fc_input=torch.permute(fc_input,(1,0,2))
        fc_input=(fc_input.squeeze(dim=0))  
        
        ## Reshape from (batch_size,1,1) to (batch_size,1)
        out=self.fc(fc_input).squeeze(dim=1)      
        return out



##### =======================================
### LSTM for forecasting 

import torch.nn.functional as F

class LSTM_with_Attention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size,
                 fc_hidden_dims, device, dropout_prob, bidirectional):
        super().__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            dropout=dropout_prob, batch_first=True,
                            bidirectional=bidirectional)

        self.dropout = nn.Dropout(dropout_prob)

        # Attention layer: project hidden outputs to attention scores
        self.attn = nn.Linear(hidden_size * self.num_directions, 1)

        # Fully connected layers
        dims = [hidden_size * self.num_directions] + fc_hidden_dims + [output_size]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.BatchNorm1d(dims[i + 1]))
                #layers.append(nn.LayerNorm(dims[i + 1])) 
                layers.append(nn.ReLU())
        layers.append(nn.Softmax(dim=1))
        self.fc = nn.Sequential(*layers)

    def forward(self, x, seq_lens, mode):
        h0 = torch.zeros(self.num_layers * self.num_directions, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers * self.num_directions, x.size(0), self.hidden_size).to(self.device)

        if mode == 'training':
            x_packed = pack_padded_sequence(x, seq_lens.cpu(), enforce_sorted=False, batch_first=True)
            output_packed, _ = self.lstm(x_packed, (h0, c0))
            output, _ = pad_packed_sequence(output_packed, batch_first=True)
        else:
            output, _ = self.lstm(x, (h0, c0))  # (batch, seq_len, hidden*dir)

        # Attention: score each time step
        attn_scores = self.attn(output).squeeze(-1)  # (batch, seq_len)
        attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(-1)  # (batch, seq_len, 1)

        # Weighted sum of outputs (context vector)
        context = torch.sum(attn_weights * output, dim=1)  # (batch, hidden*dir)

        # Feed context vector into fully connected layers
        out = self.fc(context)
        return out


##### =======================================
class LSTM_with_AttnWeightedPooling(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size,
                 fc_hidden_dims, device, dropout_prob, bidirectional,
                 n_heads):
        super().__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.lstm_output_dim = hidden_size * self.num_directions

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            dropout=dropout_prob if num_layers > 1 else 0,
                            batch_first=True, bidirectional=bidirectional)

        self.attn_layer = nn.MultiheadAttention(embed_dim=self.lstm_output_dim,
                                                num_heads=n_heads,
                                                dropout=dropout_prob,
                                                batch_first=True)

        self.dropout = nn.Dropout(dropout_prob)

        # Attention-weighted pooling layer (per timestep score)
        self.attn_pool_proj = nn.Linear(self.lstm_output_dim, 1)

        # Fully connected layers
        dims = [self.lstm_output_dim] + fc_hidden_dims + [output_size]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.BatchNorm1d(dims[i + 1]))
                #layers.append(nn.LayerNorm(dims[i + 1]))
                layers.append(nn.ReLU())
        layers.append(nn.Softmax(dim=1))  # Use sigmoid if binary classification
        self.fc = nn.Sequential(*layers)

    def forward(self, x, seq_lens, mode='training'):
        batch_size = x.size(0)

        h0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size).to(self.device)

        if mode == 'training':
            packed_input = pack_padded_sequence(x, seq_lens.cpu(), batch_first=True, enforce_sorted=False)
            packed_output, _ = self.lstm(packed_input, (h0, c0))
            lstm_out, _ = pad_packed_sequence(packed_output, batch_first=True)
        else:
            lstm_out, _ = self.lstm(x, (h0, c0))

        # Apply multi-head attention
        attn_out, _ = self.attn_layer(lstm_out, lstm_out, lstm_out)

        # Attention-weighted pooling
        attn_scores = self.attn_pool_proj(attn_out).squeeze(-1)  # (batch_size, seq_len)
        attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)  # (batch_size, seq_len, 1)
        context = torch.sum(attn_out * attn_weights, dim=1)  # (batch_size, hidden_dim)

        out = self.fc(context)
        return out


##### =======================================
## Define CustomLoss -> combine CrossEntropy loss with MSEloss
#  This helps prediction of TTP week -> predicted week:2, actual week:6, then MSEloss helps additionally to add more loss
class CrossEntropy_MSELoss(nn.Module):
    def __init__(self, weight_celoss,weight_mse_loss,ce_weight=None):
        super(CrossEntropy_MSELoss,self).__init__()
        self.ce_weight=ce_weight
        self.weight_celoss=weight_celoss
        self.weight_mse_loss=weight_mse_loss

    def forward(self, input, target):
        CEloss_funct=torch.nn.CrossEntropyLoss(weight=self.ce_weight)

        celoss=CEloss_funct(input,target)
        
        mse_input=torch.argmax(input.detach(),dim=1).to(torch.float)
        mse_target=target.to(torch.float)
        
        mseloss_funct=torch.nn.MSELoss()
        mseloss=mseloss_funct(mse_input,mse_target)
        
        celoss_weight=self.weight_celoss/(self.weight_celoss+self.weight_mse_loss)
        mseloss_weight=self.weight_mse_loss/(self.weight_celoss+self.weight_mse_loss)

        loss=celoss_weight*celoss + mseloss_weight*mseloss

        return loss

##### =======================================

### Create training dataset by creating sliding windowed-data of patients' dataframe and concatenate them
def calculate_sliding_windows(X): 
    X_id_comb_list=[]
    
    ## For each patient get the days in the study as a list and create a rolling-window style list of them
    #  i.e. Pat1 has DAY:[1,2,4,6,8,10] ->windows_size=2 ->[[1,2],[2,4],[4,6]...[8,10]]
    #  Window size is from 1 -> number of days for patient
    for pat_id in X['USUBJID'].unique()[:]:
        pat_df_X=X[X['USUBJID']==pat_id].sort_values(by=['DAY'],ascending=True)


        # Set the range of the window from 1 to the number of days
        #for window_size in range(1,len(pat_df_X['DAY']) + 1):
        for window_size in [len(pat_df_X['DAY'])]:
            # Extract the indices of combinations for the current window size 
            X_id_combination=[list(pat_df_X['DAY'].iloc[i:i+window_size].index) for i in range(len(pat_df_X['DAY']) - window_size + 1)]
             
            # Append the indices to list
            X_id_comb_list.extend(X_id_combination)

    
    ## Based on the previously created indices subset the X and y (which should be==X_train,y_train_with_index) data
    #  save them in a respective list and concatenate them in the end.
    ## Add a unique combination number to each of the sliding-window combination, so DataLoader can load the data
    #  based on this column

    X_train_list=[]

    for comb_n,X_id in zip(range(len(X_id_comb_list)),X_id_comb_list):
        X['COMBNUM']=comb_n
        X_train_list.append(X.loc[X_id,:])

    X_train_slid_wind=pd.concat(X_train_list,axis=0)    

    return X_train_slid_wind

##### =======================================
## Setup function for calculating elapsed time
def print_elapsed_time(start,stop):
    # Calculate the elapsed time in seconds
    elapsed_seconds=stop-start
    
    # Convert elapsed time to hours and minutes
    elapsed_minutes, elapsed_seconds = divmod(int(elapsed_seconds), 60)
    elapsed_hours, elapsed_minutes = divmod(elapsed_minutes, 60)
    
    # Print the result in the desired format
    return(f"{elapsed_hours} hours: {elapsed_minutes} minutes: {elapsed_seconds} seconds")

##### =======================================
def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created successfully.")


##### =======================================
# Return averaged model from saved state_dict
def return_averaged_model(model_state_dict,lstm_parameters):
    
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=LSTM_forecast(lstm_parameters['input_size'], lstm_parameters['hidden_size'], \
                                    lstm_parameters['num_of_lstm_layers'], lstm_parameters['output_size'], 
                                    lstm_parameters['fc_hidden_dims'],device,
                                    lstm_parameters['dropout_prob'],
                                    bidirectional=lstm_parameters['bidirectional'])
    
    model=AveragedModel(model,device=device,use_buffers=True)

    model.load_state_dict(model_state_dict)

    return model







####======================================================

# 2. Function for data splitting and ML model training


## Function for splitting a list of patient IDs into training and testing sublists of patient IDs-> 
#  - This function splits up the sliding-window dataset into patients who are used for training and patients 
#    whose used for testing + it also splits up the training patients IDs into K-fold training-validation sets
def stratified_train_test_split_with_CV_split(y,test_data_ratio,final_pat_ids_for_analysis,cv_repeat_num,period_end_day,
                                              k_folds,rand_state):
    import random     
    import itertools 
    from sklearn.model_selection import train_test_split, StratifiedKFold
    
    ## Set seed
    #if seed is None:
    #    random.seed()
    #if seed is not None:
    #    random.seed(seed)


    #id_list=X_slid_wind['USUBJID'].unique().tolist()
    #print(y)
    #id_list=y.index.get_level_values('USUBJID').tolist()

    ## Split patients into 
    #training_ids, testing_ids, _, _ = train_test_split(id_list, y, test_size=test_data_ratio, stratify=y,random_state=rand_state)
    
    #pat_ids_ = y.index.get_level_values('USUBJID').tolist()
    #df_=y.loc[pat_ids_].reset_index()#
    #df_['STUDYID']=df_['USUBJID'].str.split('/',expand=True)[0].values
    #y_for_strat=df_[outcome_label].astype(str) + "_" + df_['STUDYID']#.astype(str)

    ## STRATIFY ON OUTCOME LABEL & STUDYID 
    ## IF STRATIFIED PATIENT IDS WERE ALREADY CALCULATED, LOAD THEM 
    if final_pat_ids_for_analysis is not None and cv_repeat_num is not None:

        training_ids=final_pat_ids_for_analysis[period_end_day][cv_repeat_num]['X_train_ids']
        testing_ids=final_pat_ids_for_analysis[period_end_day][cv_repeat_num]['X_test_ids']
    
    else:
        ## Split patients into 
        training_ids, testing_ids, _, _ = train_test_split(pat_ids_, y, test_size=test_data_ratio, 
                                                           stratify=y,random_state=rand_state)


    ## If NO CV (==k_folds is None), return only training/testing IDs as nested listds (to keep same format as with k_folds== not None)
    #  Return training,validation, testing IDs -> testing_ids is split up in half: validation_ids (half) + testing_ids (half)
    #print('k_folds in CV-splitting function',k_folds)
    if k_folds is None:
        val_data_split_index=int(len(testing_ids)/2)
        validation_ids=testing_ids[:val_data_split_index]
        half_testing_ids=testing_ids[val_data_split_index:]
        return training_ids,testing_ids,[training_ids], [validation_ids]
        
    '''
    if k_folds==1:
        val_data_split_index=int(len(testing_ids)/2)
        validation_ids=testing_ids[:val_data_split_index]
        half_testing_ids=testing_ids[val_data_split_index:]
        return training_ids,half_testing_ids,[training_ids], [validation_ids]
    '''
    if k_folds is not None:
        if k_folds<1:
            raise ValueError(f'K-folds has to be >0!')

        if len(training_ids)<k_folds:
            raise ValueError(f'Number of K-folds higher than length ({len(training_ids)}) of training data! Lower K-fold or check length of training data.')
        
  
        if k_folds>1:
            print('within kfolds y:\n',y,'\n')
            # Create array for StratifiedKFold
            #y_train_array = np.array(y.loc[y.index.get_level_values('USUBJID').isin(training_ids)].values)
            #print('within kfolds y_train_array\n',y.loc[y.index.get_level_values('USUBJID').isin(training_ids)],'\n')
            y_train_array = y.loc[training_ids].values
            training_ids_array = np.array(training_ids)
    
            skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=rand_state)
    
            cv_training_id_array = []
            cv_validation_id_array = []
    
            #for train_index, val_index in skf.split(training_ids_array, y_train_array):
            for train_index, val_index in skf.split(training_ids_array, y_train_array):
                #y_for_strat
                train_ids = training_ids_array[train_index].tolist()
                val_ids = training_ids_array[val_index].tolist()
                cv_training_id_array.append(train_ids)
                cv_validation_id_array.append(val_ids)


            return training_ids,testing_ids,cv_training_id_array, cv_validation_id_array
    
# --------------------------------------------------------------------------------------------------------------------------
### Drop unnecessary columns from the whole dataset, one-hot encode categorical variables and return X and y dataframes 
def create_X_y_dataframe_from_raw_data(df_for_analysis,target_df):
    
    ## Drop unnecessary columns and create one-hot encoding of categorical vars
    data=df_for_analysis.drop(columns=['STUDYID','ARM'])
    
    ## Select nominal columns with categorical variables >3 unique values and convert them to one-hot encoded variables
    data_hot_encoded=pd.get_dummies(data.drop(columns=['USUBJID']))
    data_hot_encoded=pd.concat([data_hot_encoded,data['USUBJID']],axis=1)


    ## Drop target variable from predictor variables' dataframe
    #cols_to_drop=['mb_Time to Detection_STD_NUM_RESULT']
    X=data_hot_encoded.drop(columns=cols_to_drop)


    ## Create y data to predict + keep some data  (study id, arm) for plotting later 
    y=target_df.loc[target_df.index.isin(df_for_analysis['USUBJID'].unique())].copy()
    
    y_=df_for_analysis.drop_duplicates(subset=['USUBJID','STUDYID','ARM']).set_index('USUBJID')
    y[['STUDYID','ARM']]=y_.loc[y.index,['STUDYID','ARM']].values

    y=y.reset_index().set_index(['USUBJID','STUDYID','ARM'],drop=True)                                     

    return X,y


# --------------------------------------------------------------------------------------------------------------------------
### Drop unnecessary columns from the whole dataset, one-hot encode categorical variables and return X and y dataframes 
def one_hot_encode_cat_vars(df_for_analysis,cat_vars_not_to_hot_encode):

    data=df_for_analysis.drop(columns=cat_vars_not_to_hot_encode)
    hot_encoded_df=pd.get_dummies(data,dtype=int)
    X=pd.concat([df_for_analysis[cat_vars_not_to_hot_encode],hot_encoded_df],axis=1)                                   

    return X

# --------------------------------------------------------------------------------------------------------------------------

### Drop unnecessary columns from the whole dataset, one-hot encode categorical variables and return X and y dataframes 
def create_y_to_pats_considered(data,target_df):

    if isinstance(target_df,pd.Series):
        target_df=target_df.to_frame()
    
    ## Create y data to predict + keep some data  (study id, arm) for plotting later 
    y=target_df.loc[target_df.index.isin(data['USUBJID'].unique())].copy()

    y_=data.drop_duplicates(subset=['USUBJID','STUDYID','ARM']).set_index('USUBJID')
    y[['STUDYID','ARM']]=y_.loc[y.index,['STUDYID','ARM']].values

    y=y.reset_index().set_index(['USUBJID','STUDYID','ARM'],drop=True)                                     

    return y


# --------------------------------------------------------------------------------------------------------------------------
## Extract a std_scaler fitted to the non-binary numerical columns of the training data 
#  => will be used to standardise X_train and X_test data later on
def extract_non_binary_num_std_scaler(X):

    ## Select non-binary numerical columns
    numerical_cols=X.select_dtypes(exclude=['object'])
    non_binary_num_cols=[]

    for col in numerical_cols.columns:
        unique_values=np.sort(numerical_cols[col].unique())
        if np.array_equal(unique_values, np.array([0, 1]))==False:
            non_binary_num_cols.append(col)

    if 'COMBNUM' in non_binary_num_cols:
        non_binary_num_cols=[x for x in non_binary_num_cols if x!='COMBNUM']

    std_scaler=StandardScaler()
    std_scaler.fit(X.loc[:,non_binary_num_cols])
    
    return std_scaler,non_binary_num_cols

# --------------------------------------------------------------------------------------------------------------------------
## Create standardised X data: 
#  - only non-binary numerical columns are getting standardised)
#  - std_scaler=StandardScaler() fitted to the training_data
def standardise_non_binary_num_vars(X,non_binary_num_cols,std_scaler):
    ## If previously fitted std_scaler is provided, transform your dataframe with it an return only transformed data
    X.loc[:,non_binary_num_cols]=std_scaler.transform(X.loc[:,non_binary_num_cols])
    return X


# --------------------------------------------------------------------------------------------------------------------------
## Extract a std_scaler fitted to the 'DAY' column of the training data => will be used to standardise 'y' data's 'DAY' column
def extract_day_std_scaler(X):
    day_std_scaler=StandardScaler()
    day_std_scaler.fit(X.loc[:, 'DAY'].values.reshape(-1,1))
    return day_std_scaler



# --------------------------------------------------------------------------------------------------------------------------
##  As lot of temporal columns are very sparse -> keep only those that have information X>=threshold number of patients
def drop_temporal_cols_with_insuff_data(data,temp_cols,threshold):
    data_temp_cols=data.columns[data.columns.str.startswith(tuple(temp_cols))].tolist()

    data_temp_cols_non_zero_mask=data[data_temp_cols].apply(lambda x : x!=0)
    data_temp_cols_non_zero_mask['USUBJID']=data['USUBJID'].values
    data_temp_cols_non_zero_mask=data_temp_cols_non_zero_mask.set_index('USUBJID',drop=True)

    data_temp_cols_non_zero_df=data_temp_cols_non_zero_mask.groupby('USUBJID').apply(lambda x: (x.apply(lambda y: any(y==True))))
    temp_vars_avail_for_pats=data_temp_cols_non_zero_df.apply(lambda x: sum(x))
    temporal_cols_to_drop=temp_vars_avail_for_pats[temp_vars_avail_for_pats<threshold].index.tolist()

    return data.drop(columns=temporal_cols_to_drop)

# --------------------------------------------------------------------------------------------------------------------------
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

# --------------------------------------------------------------------------------------------------------------------------
def load_sliding_window_train_data(data_param_key,period_end_day):    
    ## Check if the sliding windows training datasets for X_train and y_train are already created ->
    #  If not raise error to create them    

    #fn_x='_'.join([data_param_key,'LSTM_slid_window_X_train_data'])+'.csv.gz'
    fn_x='_'.join([data_param_key,str(period_end_day),'LSTM_slid_window_X_train_data'])+'.csv.gz'
    data_dir='../data'
    fname_x=os.path.join(data_dir,fn_x)

    if os.path.exists(fname_x):
        print('Loading previously saved training datasets')
        X_train_slid_wind=pd.read_csv(fname_x,index_col=0,low_memory=False)

        if period_end_day not in ['all','baseline']:
            #print('period_end_day',period_end_day)
            #print(X_train_slid_wind['DAY'])
            X_train_slid_wind=X_train_slid_wind[X_train_slid_wind['DAY']<period_end_day]
        
        return X_train_slid_wind

    if os.path.exists(fname_x)==False:
        raise ValueError ('Sliding-window train data is missing, run s8_2_LSTM_regression.ipynb!')



### ==========================================================
### ==========================================================
def train_LSTM_model(X_slid_wind,train_ids,valid_ids,y,num_of_classes,outcome_label,
                     model_complex,columns_to_drop,grid_search_df,param_comb_num,lstm_parameters,
                    DataLoader_num_workers,pin_memory,ds_name,verbose):
    
    from tqdm import trange
    from tqdm.auto import tqdm

    mode='training'
    train_loss_list=[]
    lr_list=[]  


    
    # ---------------------------
    # Early stopping config
    # ---------------------------
    es_enabled = bool(lstm_parameters.get("early_stopping", True))
    es_patience = int(lstm_parameters.get("early_stopping_patience", 10))
    es_min_delta = float(lstm_parameters.get("early_stopping_min_delta", 0.0))
    es_warmup_epochs = int(lstm_parameters.get("early_stopping_warmup_epochs", 0))
    es_restore_best = bool(lstm_parameters.get("early_stopping_restore_best", True))

    best_metric = float("inf")          # monitoring val_loss 
    best_epoch = -1
    best_state = None                  # best model state_dict
    best_swa_state = None              # best swa_model state_dict (if used)
    bad_epochs = 0

    def _avg_loss_on_loader(model_to_eval, loader, criterion_, device_):
        """Compute average loss on a DataLoader."""
        model_to_eval.eval()
        running = 0.0
        n_batches = 0
        with torch.no_grad():
            #for inputs, targets, seq_lens in loader:
           
            for inputs, targets in loader:
                if inputs.size(0) == 1:
                    continue

                inputs = inputs.to(device_)
                targets = targets.to(device_)
                
                batch_size = inputs.size(0)
                seq_len = inputs.size(1)                
                seq_lens = torch.full((batch_size,),seq_len,dtype=torch.long,device='cpu')
                
                outputs = model_to_eval(inputs, seq_lens, mode)  # keep your signature
                loss = criterion_(outputs, targets.squeeze().long())
                running += float(loss.item())
                n_batches += 1
        model_to_eval.train()
        return (running / n_batches) if n_batches > 0 else float("inf")
    
    
    ### STANDARDISE INPUT DATA
    ## Fit StandardScaler to training data: 
    #  - subset whole data to the training patients 
    #  - fit standardScaler to training data and return scaler for the
    #    standardisation of validation and tesing datasets later
    X_train_slid_wind=X_slid_wind[X_slid_wind['USUBJID'].isin(train_ids)]
    std_scaler_train_data,non_binary_num_cols=extract_non_binary_num_std_scaler(X_train_slid_wind)
    
    
    ## Subset training data, standardise non-binary numerical columns 
    X_train_slid_wind=standardise_non_binary_num_vars(X_train_slid_wind,non_binary_num_cols,std_scaler=std_scaler_train_data)

    #print('X_train_slid_wind.columns',X_train_slid_wind.columns)
    #print('X_slid_wind max',X_slid_wind['DAY'].max())
    #print('Training data max',X_train_slid_wind['DAY'].max())
    #print('Training data shape',X_train_slid_wind.shape)
    
    ## Subset prediction labels to patients considered
    y_train=y[y.index.get_level_values('USUBJID').isin(X_slid_wind['USUBJID'].unique())]

    #print('---y_train labels---\n',y_train[outcome_label].value_counts(dropna=False),'\n')
    
    train_dataset=Grouped_Train_Dataset(X_train_slid_wind,y_train,model_complex,columns_to_drop)

    ## Set the weights of the predicted labels based on the frequency-distribution of each label
    if lstm_parameters['weight_Cross_Entropy_by_label_freq']==True:
        target_label_freq=y[outcome_label].value_counts(normalize=True)
        
        if len(target_label_freq)==2:
            target_label_freq=target_label_freq.sort_index(ascending=False)
        if len(target_label_freq)>2:
            target_label_freq=target_label_freq.sort_index(ascending=True)

        cross_entropy_weights_dict=dict(zip(target_label_freq.index.tolist(),[1/x for x in target_label_freq.values.tolist()]))
    
    if lstm_parameters['weight_Cross_Entropy_by_label_freq']==False:
        target_label_freq=y[outcome_label].value_counts(normalize=True)
        cross_entropy_weights_dict=dict(zip(target_label_freq.index.tolist(),[1 for x in target_label_freq.values.tolist()]))
        
    #print(cross_entropy_weights_dict)


    # Create data loaders
    #train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    # Define batch size
    batch_size=lstm_parameters['batch_size']
    train_loader=DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                            collate_fn=train_dataset.collate_fn,num_workers=DataLoader_num_workers,
                            pin_memory=pin_memory,persistent_workers=False)


    # ---------------------------
    ## Loading validation data

    x_for_val=X_slid_wind[X_slid_wind['USUBJID'].isin(valid_ids)]
    x_for_val=standardise_non_binary_num_vars(x_for_val,non_binary_num_cols,std_scaler_train_data)
    y_data_with_index=y[y.index.get_level_values('USUBJID').isin(valid_ids)]

    
    #print(x_for_test)
    #print('---y_test labels---\n',y_data_with_index[outcome_label].value_counts(dropna=False),'\n')
    val_dataset=Grouped_Test_Dataset_whole_data(x_for_val,y_data_with_index,model_complex,columns_to_drop,
                                                 dataset_type='validation')
    
    val_loader=DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=DataLoader_num_workers,
                            pin_memory=pin_memory,
                          collate_fn=val_dataset.collate_fn)
   

    # Initialize the model and optimizer           
    lstm_parameters['input_size']=X_train_slid_wind.drop(columns=['USUBJID','COMBNUM']).shape[1]
    lstm_parameters['input_size']=train_dataset[0][0].shape[1]
    #print('X_Training data shape',train_dataset[0][0].shape)

    # Initialize the model and optimizer
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    '''
    model=LSTM_forecast(lstm_parameters['input_size'], lstm_parameters['hidden_size'], \
                        lstm_parameters['num_of_lstm_layers'], lstm_parameters['output_size'], 
                        lstm_parameters['fc_hidden_dims'],device,
                        lstm_parameters['dropout_prob'],
                        bidirectional=lstm_parameters['bidirectional']).to(device)
    '''
    
    model=LSTM_with_Attention(lstm_parameters['input_size'], lstm_parameters['hidden_size'], \
                        lstm_parameters['num_of_lstm_layers'], lstm_parameters['output_size'], 
                        lstm_parameters['fc_hidden_dims'],device,
                        lstm_parameters['dropout_prob'],
                        bidirectional=lstm_parameters['bidirectional']).to(device)
    
    
    '''
    model=LSTM_with_AttnWeightedPooling(lstm_parameters['input_size'], lstm_parameters['hidden_size'], \
                                        lstm_parameters['num_of_lstm_layers'], lstm_parameters['output_size'], 
                                        lstm_parameters['fc_hidden_dims'],device,
                                        lstm_parameters['dropout_prob'],
                                        bidirectional=lstm_parameters['bidirectional'],
                                        n_heads=4).to(device)
    '''  

    ## Remove big data variables as they are not necessary for evaluation         
    del X_train_slid_wind,train_dataset                                            

    ## Set loss function + optimizer
    '''
    if lstm_parameters['criterion']=='CrossEntropyLoss':
        criterion=nn.CrossEntropyLoss(weight=torch.FloatTensor(list(cross_entropy_weights_dict.values())))
        
    if lstm_parameters['criterion']=='CrossEntropy_MSELoss':
        criterion=CrossEntropy_MSELoss(weight_celoss=lstm_parameters['weight_celoss'],
                                    weight_mse_loss=lstm_parameters['weight_mse_loss'],
                                    ce_weight=torch.FloatTensor(list(cross_entropy_weights_dict.values())))
    '''

    ce_w = None
    if lstm_parameters.get('weight_Cross_Entropy_by_label_freq', False) and cross_entropy_weights_dict is not None:
        ce_w = torch.tensor(list(cross_entropy_weights_dict.values()),
                            dtype=torch.float32, device=device)
    
    if lstm_parameters['criterion'] == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss(weight=ce_w)
    
    elif lstm_parameters['criterion'] == 'CrossEntropy_MSELoss':
        criterion = CrossEntropy_MSELoss(
            weight_celoss=lstm_parameters['weight_celoss'],
            weight_mse_loss=lstm_parameters['weight_mse_loss'],
            ce_weight=ce_w
        )


    if verbose==True:
        print('Number of batches/epoch:',len(train_loader))

    ## Set the swa lr_cycle_length to sync with number of batches /epoch
    lstm_parameters['swa_lr_cycle_length']=len(train_loader)
    
    ## Calculate the umber of swa updates / epoch 
    # - (swa update happening in the last 'num_swa_updates_per_epoch' number of steps per epoch)
    # - if swa_update_in_cycle_ratio is too low ==> num_swa_updates_per_epoch would be ==0  
    #   ==>set it to 1 to have at least one update / epoch
    num_swa_updates_per_epoch=int((1-lstm_parameters['swa_update_in_cycle_ratio']) *lstm_parameters['swa_lr_cycle_length'])
    num_swa_updates_per_epoch=num_swa_updates_per_epoch if num_swa_updates_per_epoch==0 else 1


    
    # CyclicLR function used in SWA only works with SGD, not wth ADAM. Therefore if there are SWA steps after pretraining, select
    # SGD as optimizer. If no SWA steps, select ADAM.
    if lstm_parameters['swa_epochs']==0:                        
        optimizer=torch.optim.Adam(model.parameters(), lr=lstm_parameters['pretrain_max_lr'],
                                    weight_decay=lstm_parameters['pretrain_weight_decay'])
        pretrain_scheduler=CosineAnnealingLR(optimizer,
                                            T_max=lstm_parameters['pretrain_epochs']*len(train_loader),
                                            eta_min=lstm_parameters['pretrain_min_lr'])   

    if lstm_parameters['swa_epochs']>0:                                                 
        optimizer=torch.optim.SGD(model.parameters(), lr=lstm_parameters['pretrain_max_lr'],
                                weight_decay=lstm_parameters['pretrain_weight_decay']) 
        
        pretrain_scheduler=CosineAnnealingLR(optimizer,
                                        T_max=lstm_parameters['pretrain_epochs']*len(train_loader),
                                        eta_min=lstm_parameters['pretrain_min_lr'])   

        swa_scheduler=CyclicLR(optimizer,cycle_momentum=True,
                                base_lr=lstm_parameters['swa_min_lr'],
                                max_lr=lstm_parameters['swa_max_lr'],
                                step_size_up=1,
                                step_size_down=lstm_parameters['swa_lr_cycle_length'])                                                                                                 
    
    num_of_epochs=lstm_parameters['pretrain_epochs'] + lstm_parameters['swa_epochs'] 

    
    # Train the model   
    model.train()    

    #epoch_bar = trange(num_of_epochs, desc="Training", leave=True)
    #epoch_bar = tqdm(num_of_epochs, desc="Training")
    #epoch_bar = tqdm(range(num_of_epochs), desc="Training", leave=True)
    
    #for epoch in epoch_bar:
    for epoch in range(num_of_epochs):


        epoch_start=time.time()
        running_loss=0 

        for n,(inputs, targets,seq_lens) in enumerate(train_loader,0):
            if inputs.size(0) == 1:
                continue  # Skip batch size of 1 to avoid BatchNorm error


            #print('inputs.shape',inputs.shape,'targets.shape',targets.shape,'seq_lens',seq_lens)
            #print('inputs',inputs)
            #print('targets',targets)
            inputs=inputs.to(device)
            targets=targets.to(device)
        
            outputs=model(inputs,seq_lens,mode)
    
            #print('inputs',inputs)
            #print('output, target','output_argmax',outputs, targets.squeeze(1).long(),torch.argmax(outputs.detach(),dim=1).numpy().flatten())
            loss=criterion(outputs, targets.squeeze().long())

            if epoch==0 and n==0 and verbose==True:
                print(f'Initial loss: {loss.item():.3f}')

            train_loss_list.append(loss.item())
            running_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()    

            ## Epochs after pretraining    
            if epoch+1 > lstm_parameters['pretrain_epochs']:   
                ## Initiate averaged model from the pretrained model                         
                if epoch+1==lstm_parameters['pretrain_epochs']+1:
                    swa_model=AveragedModel(model,device=device,use_buffers=True)

                ## Update the AveragedModel only with the last 'swa_update_in_cycle_ratio' percent of
                #  elements in the cycle
                #len(train_loader):
                if divmod(n,lstm_parameters['swa_lr_cycle_length'])[1] > int((1-lstm_parameters['swa_update_in_cycle_ratio']) *lstm_parameters['swa_lr_cycle_length']):
                    swa_model.update_parameters(model)
                    #print('Model updated')
                
                ## Update lr 
                swa_scheduler.step()
            else:
                ## Update lr
                pretrain_scheduler.step()
                swa_model=None

            ## Add lr to lr list for saving 
            lr_list.append(optimizer.param_groups[0]['lr'])


        epoch_end=time.time()
        t=print_elapsed_time(epoch_start,epoch_end)
        avg_loss = running_loss / len(train_loader)
        elapsed_time = epoch_end - epoch_start

        
        # Update tqdm epoch bar with loss and time info
        #epoch_bar.set_postfix({"Loss": f"{avg_loss:.4f}",
        #    "Epoch train. time": f"{elapsed_time:.1f}s"})
        
        
        ## Save weights of the model for plotting later
        # If folder doesn't exist, create it
        folder_path='../data/saved_lstm_class_models'
        create_folder_if_not_exists(folder_path)

        #fn=f'../data/saved_lstm_class_models/{ds_name}_{num_of_classes}_classes_epoch_{epoch+1}.pt'
        #pickle.dump(model.state_dict(),open(fn,"wb"))

        # ---------------------------
        # Validation + early stopping
        # ---------------------------
        if val_loader is not None:
            # If SWA is active, you usually care about swa_model once it exists.
            eval_model = swa_model if swa_model is not None else model
            val_loss = _avg_loss_on_loader(eval_model, val_loader, criterion, device)

            if verbose:
                elapsed = time.time() - epoch_start
                print(f"Epoch [{epoch+1}/{num_of_epochs}]  train_loss={running_loss/len(train_loader):.4f} "
                      f"val_loss={val_loss:.4f}  time={elapsed:.1f}s")

            # Early stopping decision
            if es_enabled and (epoch + 1) >= es_warmup_epochs:
                improved = (best_metric - val_loss) > es_min_delta
                if improved:
                    best_metric = val_loss
                    best_epoch = epoch
                    bad_epochs = 0

                    # Store best weights (copy to detach from graph; keep on CPU to be safe)
                    best_state = copy.deepcopy(model.state_dict())
                    if swa_model is not None:
                        best_swa_state = copy.deepcopy(swa_model.state_dict())
                else:
                    bad_epochs += 1

                if bad_epochs >= es_patience:
                    ## Update grid_seerch_df with best epoch
                    #grid_search_df.loc[param_num_comb,'best_early_stop_epoch']=best_epoch
                    if verbose:
                        print(f"Early stopping triggered at epoch {epoch+1}. "
                              f"Best val_loss={best_metric:.4f} at epoch {best_epoch+1}.")
                    break
        else:
            # No validation loader available
            if verbose:
                elapsed = time.time() - epoch_start
                print(f"Epoch [{epoch+1}/{num_of_epochs}]  train_loss={running_loss/len(train_loader):.4f} "
                      f"time={elapsed:.1f}s")

    # Restore best weights if requested and available
    if es_enabled and es_restore_best and (best_state is not None):
        model.load_state_dict(best_state)
        if swa_model is not None and (best_swa_state is not None):
            swa_model.load_state_dict(best_swa_state)

    ## If early stopping was not triggered, set the last epoch as best epoch
    if best_epoch == -1:
        best_epoch = copy.deepcopy(epoch)


        '''
        if verbose==True:
            ## Print epoch number
            if num_of_epochs>10:
                if epoch==0:
                    print(f'Epoch [{epoch+1}/{num_of_epochs}], Loss: {loss.item():.3f}, Epoch runtime: {t}' )
                if (epoch+1) % round(num_of_epochs/10) == 0:
                    print(f'Epoch [{epoch+1}/{num_of_epochs}], Loss: {loss.item():.3f}, Epoch runtime: {t}')
            else:
                print(f'Epoch [{epoch+1}], Loss: {loss.item():.7f}, Epoch runtime: {t}')     
        '''
    ## Add training loss + learning rates
    norm_train_loss_list=[tr_loss/max(train_loss_list) for tr_loss in train_loss_list]

    return model,swa_model,train_loss_list,lr_list,std_scaler_train_data,non_binary_num_cols,best_epoch

# --------------------------------------------------------------------------------------------------------------------------
### Traing the models
def train_models(data,
                 target_df_,
                 pat_ids_,
                 period_end_day,
                 cat_vars_not_to_hot_encode,
                 scoring_methods,
                 outcome_label,
                 training_days,
                 random_state,
                 data_split_random_state,
                 test_data_results,
                 train_data_results,
                 parameter_sets,
                 ds_name,
                 cross_entropy_weights_dict,
                 model_complex,
                 k_folds,
                 best_param_combinations,
                 start,
                  categorical_map_dict,
                  num_of_cv_repeats,
                  columns_to_drop,
                 grid_search_df,
                 final_pat_ids_for_analysis,
                 verbose,
                 cross_validation=False, 
                 get_feature_importances=False):

           

    from tqdm import tqdm,trange
    import copy
    
    ## Print elapsed time since the start
    #training_start=time.time()
    #t=print_elapsed_time(start,training_start)
    #print('Elapsed time since start:',t)

    ## Num of classes to predict
    num_of_classes=len(categorical_map_dict['labels'])

    ## Create dataframe containing hyperparameters
    grid_search_df=pd.DataFrame.from_dict(parameter_sets).T
    grid_search_df.index.name='param_combnum'

        
    ##### TRAINING #######   

    
    ### INITIALIZE LSTM MODEL BASIC PARAMETERS ###

    ## Init device -> if CUDA, set pin_memory=True in order to make memory access faster
    #  https://saturncloud.io/blog/what-is-pytorch-and-how-does-pinmemory-work-in-dataloader/#:~:text=Pinned%20memory%20is%20a%20special,could%20with%20regular%20CPU%20memory.
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        pin_memory=True
    if not torch.cuda.is_available():
        pin_memory=False

    # Create data loaders
    #train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    DataLoader_num_workers=0

    ##### LOAD SLIDING-WINDOW TRAINING DATA #######
    ## Check if the sliding windows training datasets for X_train and y_train are already created ->
    #  If not raise error to create them    
    X_slid_wind=load_sliding_window_train_data(ds_name,period_end_day)
    
    ## Save columns of sliding_window training data -> at testing these colnames can be used to subset testing data
    #  columns -> costly drop_temporal_cols_with_insuff_data() function doesn't have to be run again
    X_slid_wind_cols=X_slid_wind.drop(columns=['COMBNUM']).columns


    ## One-hot-encode categorical variables, except for the pre-defined ones  in "cat_vars_not_to_hot_encode"
    X_slid_wind=one_hot_encode_cat_vars(X_slid_wind,cat_vars_not_to_hot_encode)
    
    ## Subset prediction target dataframe to the patients considered
    #  + add some information (STUDYID, ARM) to the index of y
    y=create_y_to_pats_considered(data,target_df_)


    ##### TRAIN-TEST SPLIT DATA #######
    ## Split the sliding window data (which contains all the data converted
    #  to sliding-window style) into training-validation pairs for CV + testing data. 

    ## Set CV parameters
    if cross_validation==False:
        k_folds_=1
        #num_of_train_reps=['training_results_no_CV']
        num_of_train_reps=['cv_rep_'+str(cv_rep+1) for cv_rep in range(num_of_cv_repeats)]  
    if cross_validation==True:
        num_of_train_reps=['cv_rep_'+str(cv_rep+1) for cv_rep in range(num_of_cv_repeats)]    
        k_folds_=k_folds

    ## Collect trainig results into cv_rep_dict, which will be added in the end to "train_data_results"
    cv_rep_dict={}
    #for rep_num,rep in enumerate(num_of_train_reps):
    for rep_num,rep in  tqdm(enumerate(num_of_train_reps), desc="Outer-CV-repeat", leave=False):

        #if cross_validation==True:
        print(f'\n===={ds_name} - {period_end_day} - {rep}/{num_of_cv_repeats} CV repeats===\n')

        ## Print elapsed time since the start
        training_start=time.time()
        t=print_elapsed_time(start,training_start)
        print('Elapsed time since start:',t)                    

        test_data_results[rep]={}       
        cv_rep_dict[rep]={}

        ## 'k' is iterating with CV-repeats  => 'k' makes sure, that the data is split differently among
        #  CV-repeats, but the same way between tran_ds_types ('raw','log')
        data_split_random_state_=data_split_random_state + rep_num

        ## STRATIFY ON OUTCOME LABEL & STUDYID 
        ##. ==> WITHIN STUDY ROC-AUC CLAUCLATION IS POSSIBLE, AS THERE ALWAYS WILL BE AT LEAST ONE UNFAVOUR. LABEL FROM BOTH STUDIES IN THE TEST SET
        df__=target_df_.reset_index().drop_duplicates(subset='USUBJID')#.set_index('USUBJID',drop=True)
        df__['STUDYID']=df__['USUBJID'].str.split('/',expand=True)[0].values
        df__=df__.set_index('USUBJID')
        y_for_strat=df__[outcome_label].astype(str) + "_" + df__['STUDYID']#.astype(str)

        ## Split into training-validation-testing IDs, stratified by y categories
        training_pat_ids,testing_pat_ids,\
            cv_training_ids, cv_validation_ids=stratified_train_test_split_with_CV_split(y=y_for_strat.loc[pat_ids_],
                                                                                         test_data_ratio=0.2,
                                                                                         final_pat_ids_for_analysis=final_pat_ids_for_analysis,
                                                                                         cv_repeat_num=rep_num,
                                                                                         period_end_day=period_end_day,
                                                                                         k_folds=k_folds,
                                                                                         rand_state=data_split_random_state_)
        cv_rep_dict[rep]['training_pat_ids']=training_pat_ids
        cv_rep_dict[rep]['validation_pat_ids']=list(set(chain(*cv_validation_ids)))
        cv_rep_dict[rep]['testing_pat_ids']=testing_pat_ids

        cv_rep_dict[rep]['results']={'training':{},'validation':{}}
        cv_rep_dict[rep]['training_loss']={}
        cv_rep_dict[rep]['training_learning_rate']={}
        
        cv_rep_dict[rep]['cv_training_ids']=cv_training_ids
        cv_rep_dict[rep]['cv_validation_ids']=cv_validation_ids 
        #cv_rep_dict[rep]['results']={}#
        

        ## IF PARAMETER SEARCH IS PERFORMED, LOOP OVER ALL PARAMETER COMBINATIONS
        if best_param_combinations is None:
            parameter_sets_=copy.deepcopy(parameter_sets)

        ## IF FINAL MODEL IS TRAINED, ONLY LOOP OVER THE BEST PARAMETER COMBINATION
        if best_param_combinations is not None:
            parameter_sets_={0:best_param_combinations[rep]}
            print('best params:',parameter_sets_)
    
        
        ##  - FOR EACH HYPERPARAMETER COMBINATION, RUN THE TRAINING K_FOLD TIMES, if parameter_search is done,
        ##. - If final training, then best parameter combination for given CV-rep is used for training
        #for param_comb_num,lstm_parameters in parameter_sets_.items():
        for param_comb_num, lstm_parameters in tqdm(parameter_sets_.items(), desc="Hyperparameter combinations",leave=False):

            #print()
            test_data_results[rep]['results']={}
            test_data_results[rep]['results'][param_comb_num]={}

            cv_rep_dict[rep]['results']['training'][param_comb_num]={}
            cv_rep_dict[rep]['results']['validation'][param_comb_num]={}
            #cv_rep_dict[rep]['results']['training'][param_comb_num]['training_loss']={}
            #cv_rep_dict[rep]['results']['training'][param_comb_num]['training_learning_rate']={}
            #cv_rep_dict[rep]['results']['training'][param_comb_num]['model']={}

            
            ## Run training K-fold times (if no CV -> just 1 loop with training and testing data)
            #for k, train_ids,valid_ids in zip(range(k_folds_),cv_training_ids,cv_validation_ids):

            for k, train_ids,valid_ids in tqdm(zip(range(k_folds_),cv_training_ids,cv_validation_ids),
                                               desc="Inner-CV",leave=True):
                
                if cross_validation==True and verbose==True:
                    print(f'Starting training of {k+1}/{k_folds_}-fold\n')

                cv_rep_dict[rep]['results']['training'][param_comb_num][k]={}
                cv_rep_dict[rep]['results']['validation'][param_comb_num][k]={}
                

                

                ### TRAIN LSTM MODEL
                model,swa_model,train_loss_list,\
                    lr_list,std_scaler_train_data,\
                        non_binary_num_cols,best_epoch = train_LSTM_model(X_slid_wind,train_ids,valid_ids,
                                                                          y,num_of_classes,outcome_label,
                                                                         model_complex,columns_to_drop,
                                                                         grid_search_df,param_comb_num,
                                                                         lstm_parameters,DataLoader_num_workers,
                                                                         pin_memory,ds_name,verbose)

                ## Add training loss + learning rates
                norm_train_loss_list=[tr_loss/max(train_loss_list) for tr_loss in train_loss_list]
                #plt.plot(np.arange(len(train_loss_list)),norm_train_loss_list,label=f'{rep_num}-split/{k}-fold')
                #plt.plot(np.arange(len(train_loss_list)),train_loss_list,label=f'{rep_num}-split/{k}-fold')
                #plt.legend()
                #plt.ylabel('Normalized training loss')
       
                cv_rep_dict[rep]['results']['training'][param_comb_num][k]['training_loss']=train_loss_list
                cv_rep_dict[rep]['results']['training'][param_comb_num][k]['training_learning_rate']=lr_list
                cv_rep_dict[rep]['results']['training'][param_comb_num][k]['best_epoch']=best_epoch
                
                ## Set the averaged model as model for testing
                #if 'swa_model' in locals():
                if swa_model is not None:
                    model=swa_model
                    print('Using swa model as final model')
                #cv_rep_dict[rep]['results']['training'][param_comb_num][k]['model']=model
                
  
                
                training_stop=time.time()
                t=print_elapsed_time(training_start,training_stop)
                if verbose==True:
                    print('Training duration:',t,' ==> starting to test model')
                    print('-------------')                          
    
                #### TESTING THE MODEL ON TRAINING + VALIDATION SET ######
                # Evaluate the model  
                model.eval() 

                with torch.no_grad():
                    mode='testing'  
                    batch_size=lstm_parameters['batch_size']
    
                    ## IF CV is true, evaluate the internal validation sets as well, if there is no CV, just evaluate the training data
                    if cross_validation==True:
                        cohort_list=['training','validation']
                    
                    if cross_validation==False:
                        cohort_list=['training']
                        
                    ## Run prediction with trained model on Training and Testing dataset
                    for dataset_type in cohort_list:
    
                        if dataset_type=='training':
                            x_for_test=X_slid_wind[X_slid_wind['USUBJID'].isin(train_ids)]
                            x_for_test=standardise_non_binary_num_vars(x_for_test,non_binary_num_cols,std_scaler_train_data)
                            y_data_with_index=y[y.index.get_level_values('USUBJID').isin(train_ids)]                           
                            
                                                                                    
                        if dataset_type=='validation':
                            x_for_test=X_slid_wind[X_slid_wind['USUBJID'].isin(valid_ids)]
                            x_for_test=standardise_non_binary_num_vars(x_for_test,non_binary_num_cols,std_scaler_train_data)
                            y_data_with_index=y[y.index.get_level_values('USUBJID').isin(valid_ids)]
      
    
                        #print(x_for_test)
                        #print('---y_test labels---\n',y_data_with_index[outcome_label].value_counts(dropna=False),'\n')
                        test_dataset=Grouped_Test_Dataset_whole_data(x_for_test,y_data_with_index,model_complex,columns_to_drop,
                                                                                dataset_type=dataset_type)
    
                        test_loader=DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                                num_workers=DataLoader_num_workers,
                                                pin_memory=pin_memory,collate_fn=test_dataset.collate_fn)
                        
                        predicted_output_label_list,predicted_output_prob_list,target_list=[],[],[]
                        for inputs, targets in test_loader:
                        
                            #print('test input shape',inputs.squeeze(0).shape)
                            #print('test targets',targets.detach().cpu().numpy())
                            inputs = inputs.to(device)
                            targets = targets.to(device)
                            batch_size = inputs.size(0)
                            seq_len = inputs.size(1)                
                            seq_lens = torch.full((batch_size,),seq_len,dtype=torch.long,device='cpu')
                        
                            predicted_output=model(inputs,seq_lens,mode)
                            #print('predicted_output',predicted_output.detach().cpu().numpy()[:,].shape,'targets',targets.detach().cpu().numpy().shape)#squeeze(1))
                            predicted_output_prob_list.append(predicted_output.detach().cpu().numpy()[:,1])
                            predicted_output_label_list.append(torch.argmax(predicted_output.detach(),dim=1).cpu().numpy().flatten())
                            target_list.append(targets.detach().cpu().numpy().squeeze(1))
                        
                        predicted_output=list(chain(*predicted_output_label_list))
                        predicted_output_probs=list(chain(*predicted_output_prob_list))
                        targets=list(chain(*target_list))
                        #print('predicted_output_probs',predicted_output_probs)
                        #print('targets',targets)
                        #print(f'Number of {dataset_type} predicted outputs',len(predicted_output))

                        sample_weights=np.vectorize(cross_entropy_weights_dict.get)(predicted_output)
                        roc_auc=roc_auc_score(y_true=targets,
                                                y_score=predicted_output_probs,
                                                sample_weight=sample_weights,
                                                average='macro')
                        
                        #roc_auc=roc_auc_score(targets, predicted_output_probs)
                        if verbose==True:
                            print(dataset_type +f' ROC-AUC score: {(roc_auc):.4f}')
                            print('========') 


                        #cv_rep_dict[rep]['results'][dataset_type][param_comb_num][k]={}
                        cv_rep_dict[rep]['results'][dataset_type][param_comb_num][k]['y']=targets
                        cv_rep_dict[rep]['results'][dataset_type][param_comb_num][k]['y_with_index']=y_data_with_index
                        cv_rep_dict[rep]['results'][dataset_type][param_comb_num][k]['predicted_label']=predicted_output
                        cv_rep_dict[rep]['results'][dataset_type][param_comb_num][k]['predicted_prob']=predicted_output_prob_list
                        cv_rep_dict[rep]['results'][dataset_type][param_comb_num][k]['cross_entropy_weights_dict']=cross_entropy_weights_dict
                        #cv_rep_dict[rep]['results'][dataset_type][param_comb_num][k]['x']=x_for_test
                        cv_rep_dict[rep]['results'][dataset_type][param_comb_num][k]['roc_auc']=roc_auc
                        #cv_rep_dict[rep]['results']['training'][k]['model_state_dict']=model.state_dict()

            ## Average over the early stopping best epoechs for each  inner CV-fold (k), and save it as the best epoch to stop for given
            #. parameter combination tested
            param_comb_d = cv_rep_dict[rep]['results']['training'][param_comb_num]
            avg_best_epoch = np.mean([param_comb_d[k]['best_epoch'] for k in param_comb_d.keys()])
            grid_search_df.loc[param_comb_num,'early_stopping_best_epoch']=avg_best_epoch
            cv_rep_dict[rep]['grid_search_df'] = grid_search_df
            
 
            ###====== TRAINING FINAL MODEL =====#### 
    
            ## Only test final model:
            #  - If final model is trained (best_param_combinations is not None) \
            #  - OR there is no CV performed (k_folds is None)
            if best_param_combinations is not None or k_folds is None:
               
        
                ## Create an AveragedModel object using the first model trained
                #  ==> if there was no CV, this model is the final model!!
                #model_ = cv_rep_dict[rep]['results']['training'][param_comb_num][0]['model']
                #final_model = cv_rep_dict[rep]['results']['training'][param_comb_num][0]['model']
                #final_model=AveragedModel(model_,device=device,use_buffers=True)
                

                '''
                ## If there were multiple k_folds, TRAIN FINAL MODEL ON ALL THE TRAINING PATIENTS OF GIVEN CV-SPLIT
                if k_folds_>1:
    
                    print('Training final model')
                    final_model,swa_model_,_,\
                            _,_,_ = train_LSTM_model(X_slid_wind,training_pat_ids,y,
                                                     num_of_classes,model_complex,columns_to_drop,
                                                     lstm_parameters,DataLoader_num_workers,
                                                      pin_memory,ds_name,verbose)
                    if swa_model_ is not None:
                        final_model=swa_model_
                        
                    
                    ## AVERAGE THE MODELS'WEIGHTS FROM THE K-FOLD TRAINING INTO ONE FINAL MODEL
                    DEPRECATED: avergage model weights across the k models that have been trained on the internal K-fold splits
                    for k_ in range(1,k_folds_):
                        model_=cv_rep_dict[rep]['results']['training'][param_comb_num][k_]['model']
                        final_model.update_parameters(model_)
                '''
    
                #### TESTING THE MODEL ON HELD-OUT TESTING SET ######
                with torch.no_grad():
                    mode='testing'
                    cv_rep_dict[rep].pop('model', None)
                    #test_data_results[rep]['final_model_state_dict']=final_model.state_dict()
                    
                    x_for_test=X_slid_wind[X_slid_wind['USUBJID'].isin(testing_pat_ids)]
                    x_for_test=standardise_non_binary_num_vars(x_for_test,non_binary_num_cols,std_scaler_train_data)
                    y_data_with_index=y[y.index.get_level_values('USUBJID').isin(testing_pat_ids)]
        
        
                    test_dataset=Grouped_Test_Dataset_whole_data(x_for_test,y_data_with_index,model_complex,columns_to_drop,
                                                                    dataset_type=dataset_type)
        
                    test_loader=DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                            num_workers=DataLoader_num_workers,
                                            pin_memory=pin_memory,collate_fn=test_dataset.collate_fn)
        
                    predicted_output_label_list,predicted_output_prob_list,target_list=[],[],[]
                    for inputs, targets in test_loader:

                        inputs = inputs.to(device)
                        targets = targets.to(device)
                        batch_size = inputs.size(0)
                        seq_len = inputs.size(1)                
                        seq_lens = torch.full((batch_size,),seq_len,dtype=torch.long,device='cpu')
                    
                        predicted_output=model(inputs,seq_lens,mode)
                        predicted_output_prob_list.append(predicted_output.detach().cpu().numpy()[:,1])
                        predicted_output_label_list.append(torch.argmax(predicted_output.detach(),dim=1).cpu().numpy().flatten())
                        target_list.append(targets.detach().cpu().numpy().squeeze(1))
                    
                    predicted_output=list(chain(*predicted_output_label_list))
                    predicted_output_probs=list(chain(*predicted_output_prob_list))
                    targets=list(chain(*target_list))

                    sample_weights=np.vectorize(cross_entropy_weights_dict.get)(predicted_output)
                    roc_auc=roc_auc_score(y_true=targets,
                                            y_score=predicted_output_probs,
                                            sample_weight=sample_weights,
                                            average='macro')
                    
                    #roc_auc=roc_auc_score(targets, predicted_output_probs)
                    if verbose==True:
                        print('Testing' + f' ROC-AUC score: {(roc_auc):.4f}')
                        #print('len(test targets)',len(targets))
                        print('========')    
        
                    
                    test_data_results[rep]['results'][param_comb_num]['y']=targets
                    test_data_results[rep]['results'][param_comb_num]['y_with_index']=y_data_with_index
                    test_data_results[rep]['results'][param_comb_num]['predicted_label']=predicted_output
                    test_data_results[rep]['results'][param_comb_num]['predicted_prob']=predicted_output_prob_list
                    test_data_results[rep]['results'][param_comb_num]['cross_entropy_weights_dict']=cross_entropy_weights_dict
                    test_data_results[rep]['results'][param_comb_num]['x']=x_for_test    
                    test_data_results[rep]['results'][param_comb_num]['roc_auc']=roc_auc
                    test_data_results[rep]['results'][param_comb_num]['model_state_dict']=final_model.state_dict()
    
    
    ## Add training (or CV) results to training_data_dict
    train_data_results=cv_rep_dict 
    
       
    return  train_data_results,test_data_results   




#### ==========
## Extract the combination of hyperparameters for each CV-split, that had the highest average ROC-AUC across the internal CV-splits
def extract_best_param_comb(result_dict):

    train_data_results = result_dict['train_data_results']
    best_param_combinations={}
    
    for rep in [*train_data_results][:]:
        grid_search_df = train_data_results[rep]['grid_search_df']
        #print('rep',rep)
    
        #best_param_combinations[rep]={}
    
        for dataset_type in ['training','validation'][1:]:
            #print(dataset_type)
    
            #cv_roc_auc_scores[rep][dataset_type]={}
            avg_roc_auc_list=[]
            
            for param_comb_num in train_data_results[rep]['results'][dataset_type].keys():
                
                #print('param_comb_num',param_comb_num)
                roc_auc_list=[]
                
                for k in train_data_results[rep]['results'][dataset_type][param_comb_num].keys():
                    
                    #print(train_data_results[rep]['results'][dataset_type][param_comb_num][k].keys())
                    roc_auc=train_data_results[rep]['results'][dataset_type][param_comb_num][k]['roc_auc']
                    roc_auc_list.append(roc_auc)
                    #print('k',k,roc_auc)
                    
                #print('mean roc_auc',np.mean(np.array(roc_auc_list)))
                avg_roc_auc_list.append(np.mean(np.array(roc_auc_list)))
            
            param_comb_num_max = np.argmax(avg_roc_auc_list)
            #print(param_comb_num_max)
            best_param_combinations[rep] = grid_search_df.loc[param_comb_num_max,:].to_dict()    
    
    return best_param_combinations

####======================================================
# 3. Functions for data preprocessing


###========================
def return_race_dict():
    data=load_merged_data_of_lab_vars()
    race_df=data.drop_duplicates(subset=['USUBJID'])[['USUBJID','STUDYID','RACE']]
    race_df.loc[race_df['STUDYID']=='TB-1022','RACE']='BLACK'
    race_dict=race_df.set_index('USUBJID')['RACE'].to_dict()

    return race_dict


##=========================================
## 
def backward_fill_and_extract_vars_at_baseline(X_subset,columns_to_drop):

    ## Extract columns selected for input 
    final_cols=X_subset.drop(columns=columns_to_drop).columns.tolist()
    
    ## Drop visits (==rows), where there are missing data in the selected variable columns
    a=X_subset[['DAY']+final_cols].dropna(how='any',subset=final_cols,axis=0)

    #print(a.columns)
    #print(a.head())
    
    ## Extract the first day of study, where the patient has no missing information in the input variables
    ## ==> this way we can check which was the first visit where all of the selected variables have non-missing measurements 
    #. ==> To impute missing variables at earlier visits, backward fill the missing variable's first valid value
    #. ==> The upper limit where a backward fill is acceptable is 31 days.
    first_complete_day_df=a.sort_values(by=['DAY']).groupby('USUBJID').apply(lambda x: x.loc[x.index[0],:])
    
    pats_with_miss_vars=first_complete_day_df[(first_complete_day_df['DAY']>-10) &((first_complete_day_df['DAY']<31))].index
    
    X_subset_=X_subset.copy()

    ## Loop over patients who have missing data in their early visits, and backward fill missing data
    for pat in pats_with_miss_vars[:]:
        pat_df=X_subset[X_subset['USUBJID']==pat]
    
        ## Get columns which have NaNs in the final columns containing input variables
        nan_cols_bool=pat_df.loc[:,['DAY']+final_cols].sort_values('DAY').isna().any(axis=0)
        nan_cols=nan_cols_bool[nan_cols_bool.values].index.tolist()#+ ['dr_reg_study_drugs_cumul']
    
        ## backward fill those columns with the first observed value + insert backward filled data into the X_subset dataframe
        interp_df=pat_df[nan_cols].interpolate('bfill')
        X_subset_.loc[interp_df.index,nan_cols] = interp_df.values
    
    ## Get all the visits before DAY 5 of the study and drop those visits, where despite of the backward filling, there are still NaNs
    c=X_subset_[['DAY']+final_cols].sort_values(by=['DAY']).groupby('USUBJID').apply(lambda x: x.loc[(x['DAY']<5),:]).dropna(how='any',subset=final_cols,axis=0)#['USUBJID'].unique().shape
    
    ## If there are multiple early visits (usually in the week prior to therapy start, and then on the first 1-5 days), take the earliest timepoint as the baseline
    #. + drop drug regimen data columns, as theoretically no drugs have been taken yet
    #  + drop cumulate adverse events columns, as there was no therapy 
    X_subset_baseline=c.drop(columns=['USUBJID']).groupby('USUBJID',as_index=True).apply(lambda x: x.loc[(x.index[0]),:]).drop(columns=c.columns[c.columns.str.contains('dr_reg|drugs_cumul|cumul_toxgrade')])#.dropna(how='any',subset=final_cols,axis=0)#['DAY']
    #X_subset_baseline['index']=np.nan
    
    return X_subset_baseline.reset_index()#.drop(columns='')




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


    if len(pats_with_treatment_gaps)>0:

        ## For patients with treatment gaps, extract the first day after the treatment gap
        print(dr_reg.loc[dr_reg['USUBJID'].isin(pats_with_treatment_gaps),['DAY','USUBJID']].groupby('USUBJID',as_index=True).apply(lambda x:\
                                                                                                                       x.loc[x['DAY'].diff()>1,'DAY']))
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
def subset_pats_with_therapy_in_period(period_num,period_end_days):
    
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
        race_dict = return_race_dict()
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

    ## One-hot encode 'ARM'
    if 'with_arm' in data_param_key:# and period_end_day!='baseline':
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
        

    return X,race_colnames


###========================
def load_merged_data_of_lab_vars():
    #load patient IDs who are considered in this  analysis
    pat_id_df=pd.read_csv('../data/patients_in_analysis.csv.gz',index_col=0)
    # get all pat ids
    all_ids=pat_id_df['USUBJID'].to_list()

    fname='merged_df.csv.gz'
    
    fn=os.path.join('../data/',fname)
    merged_df=pd.read_csv(fn,low_memory=False,index_col=0)

    return merged_df



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




#####=======================================
def return_predict_label_dataframe(parameters_for_analysis,data_param_key,X,
                                  outcome_df,outcome_label,model_names=None):
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
        outcome_label_ = copy.deepcopy(outcome_label)
        
    if parameters_for_analysis[data_param_key]['result_cat']=='RELAPSE':
        #outcome_label='RELAPSE'
        pats_with_relapse_df=extract_21_22_relapse_pats()

        pats_with_relapse_df = pats_with_relapse_df.loc[list(set(X['USUBJID'].unique())&set(pats_with_relapse_df.index))]

        ## Create new prediction labels (or even multilabels) in the "RELAPSE" column based on the relapse day intervals defined in "bins" 
        if 'bins' in parameters_for_analysis[data_param_key].keys():
            pats_with_relapse_df = cut_relapse_days_to_interval_categories(pats_with_relapse_df,data_param_key)
        
        pat_ids=pats_with_relapse_df.index.tolist()
        target_df=pats_with_relapse_df[[outcome_label]]
        y=pats_with_relapse_df[[outcome_label]] 
        outcome_label_ = copy.deepcopy(outcome_label)

    '''
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
        
    '''    
    return pat_ids,y,target_df,outcome_label_#,clust_df_concat

