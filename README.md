**This Readme describes the overall structure of the outcome_prediction repository**


# General overview
- This description provides a general overview and structure of the notebooks and python files in this repo. Each notebook contains more detailed descriptions of its steps

- The repo can be split into multiple sections:

## Notebooks s0 to s4: preprocessing
- **s0**:
    - notebook subsetting patients to individuals with available phase 3 outcome labels
- **s1**:
    - subsetting preprocessed data from the repo preprocessing all TB-PACTS data (version 2021 August, https://github.com/gargerd/TBPACTS_preprocessing) to the data of the patients with available phase 3 outcome labels ==> requires the output of the TBPACTS_preprocessing repo!
- **s2-s4**:
    - notebooks performing further data preprocessing by dataset type, and yielding one final dataframe per dataset type
    - Dataset types: 
        - dm: Demographic descriptors
        - mb: Microbiological test results
        - vs: Vital signs
        - re: Chest X-ray findings
        - lb: Laboratory test results
        - dr_reg: Cumulative drug doses taken
        - mh: Medical history
        - ms: Microbiological susceptibility
        - cmdos: Cumulative concomitant medication taken
        - cmind: Indication of why concomitant medication was taken (not used in analysis)
        - cm: Concomitant medication taken
        - ce: Clinical events
        - su: Substance use
        - ae: Adverse events 

## Notebooks s5_1 to s5_3: concatenation of individual dataset type dataframes 
- **s5_1: Extract variable availability for patients**
    - creating booelan dataframe indicating variable availability for patients, used to select which variables to subset to per patient in **s5_2**
- **s5_2: Load patient data into nested dictionaries**
    - for each patient, load their individual dataset type dataframe into nested dictionaries and save them as input to the next step
- **s5_3: Merge dataframes of dataset types into one dataframe**
    - Merge the dataframes from the nested dictionaries from s5_2 of the different dataset types into one "master" dataframe
    - Include dataset types, that were measured at some or all clinical visits during therapy period, and have a tractable number of individual variables (dm, mb, vs, re, lb, ms, ce, su)
    - Dataset types with values originating from one timepoint with thousands of individual variables (mh), or dataset types with daily values during therapy (dr_reg),  or daily values during therapy +  thousands of individual variables (cm, cmdos, ae) will be added at later timepoint
- **If any changes made in notebooks s0-s4, rerun s5_1 - s5_3 (in that order) for the changes to be incorporated in the downstream analysis (notebooks >s7)**


## Notebook s6: outcome label extraction
- Extracts outcome data labels per patient for the studies TB-1018, TB-1020, TB-1021, TB-1022, TB-1030
- Returns multiple outcome labels: outcome at end-of-therapy, at 12/18 months after start-of-therapy, 24 months after end-of-therapy, etc..


## Notebooks s7_0 to s7_4: Preprocess data for ML
- **s7_0: Functions for s7_2**
    - Contains functions for **s7_2**, which imports all the function of s7_0 in its first cell
    - **IMPORTANT**: The last cell of s7_0 has functions extracting relapse information of patients. These have to stay identical to the same functions extracting relapse information in s9_3!
- **s7_1: Extract variable availability (during therapy period) for patients**
    - same as **s5_1**, creating booelan dataframe indicating variable availability for patients, however only variables measured in the therapy period (baseline-mont 4 or month 6) are considered for creating the boolean dataframe in this case. It is used in **s7_2** for variable and patient selection 
- **s7_2: Create different sets of patients with common variables for analysis**
    - Cluster the boolean variable availability created in **s7_1** from both directions (patient clusters and variable clusters) for multiple prediction labels (end-of-therapy outcome, relapse, etc), and use these clusters to create sets of patients with common variables from the patient clusters, or variables with common patients from the variable clusters.
    - From these sets, select one or multiple patient-variable sets for final analysis for a given prediction label, and create two types of dataframes:
        - 1. Dataframe containing data of the selected patients and the selected variables only, with imputation performed where necessary (later to be used for the common variable experimental setup)
        - 2. Dataframes containing data of the selected patients, but including all available dataset type variables - **DEPRECATED AS SIMILAR DATAFRAME CREATED IN S7_4 IS USED FOR THE _ALL VARIABLES_ SETUP**
    - For details check last 3 cells of **s7_2** describing the selection process
- **s7_3: Visualising number of patients**
    - Visualising the number of patients available for each prediction label, and check common adverse events - **can be considered DEPRECATED, as these plots are not that relevant**
- **s7_4: Convert tabular data into string sentences for LLM input**
    - For each patient-variable set created in **s7_2**, create a dataframe containing all variables from all dataset types for the patients selected. These dataframes contain the substring '_all_data_concat.csv.gz' in them
    - For each monthly period from baseline to month 4 or 6, convert the tabluar patient data into string sentences for LLM-embedding
        - Three experimental setups:
            - _baseline_last_day_: common variables - last-visit before period cutoff; common variable, imputed dataframes are used 
            - _baseline_vars_: common variables - all "last-visit-before-period-cutoff" visits up to period cutoff (i.e. period cutoff is month 4, then the individual _baseline_last_day_ dataframes from baseline - last visit at month 1 - last visit at month 2 - last visit at month 3 are concatenated; for this the same common variable, imputed dataframes are used as for _baseline_last_day_ setup, **however they are only created in s9_3, so run the cell _CREATE TRAIN-TEST PATIENT SPLITS_ in s9_3 first in order to successfully convert this experiment's data into sentences!**)
            - _all_days_: all variables - all visits up to period cutoff; for this the dataframes containing all variables and **without imputation** are used (dataframes created in the beginning of the notebook, containing '_all_data_concat.csv.gz' substring in their names)
        - Per patient, convert the tabular data within a dataset type to a string sentence, and save them in dictionaries per dataset type
        - Per dataset type, add a prefix describing the given dataset type (i.e., mb: Microbiological test results of patient), and concatenate all sentences of a patient with their respective prefixes into one input string, that will be used for embedding

## Notebooks s8_4 to s9_4: Train & test ML models; run SHAP-analysis for raw-data models; run Kaplan-Meier analysis 
- **IMPORTANT**:
    - Most of the notebooks (**s8_4, 9_2, 9_3, 9_4**) have cells with extremely long run times. The content of these cells were therefore copied to respective .py files in order to perform their tasks using parallel SLURM computing (SLURM array jobs), speeding things up. The content of these cells and the respective .py files should be nearly identical (small differences can arise in the parameters of some functions, as function in the notebooks can use global variables defined in previous cells, and these variables need to be specifically defined in the .py files; but the calculations are identical!)
    - These .py files are executed using .slurm scripts --> **IMPORTANT: in the current version, a specific python within a given environment (_scarches_) is executing the .py files. The specific path therefore needs to be adjusted if a different environment or a different HPC is intended to be used**
    - The slurm scripts sometimes use .tsv files to pair up the array job's _SLURM_TASK_ID_ with model input parameter combinations, sometimes the input parameters are defined on the .slurm files themselves; the description below indicates which case applies to which notebook -->**IMPORTANT: check if SLURM-based HPC is also loading anaconda versions with `module` command, adjust this if necessary!**
      
- **s8_4: Raw data - LSTM model**
    - Creates dataframes with a sliding-window approach for training LSTM models, i.e. use all the visits available before each period cutoff (this can vary across patients, as they don't all have equal number of visits)
    - Creates a .tsv file containing a dataframe with the hyperparameter-combinations for the grid search pairing them up with _SLURM_TASK_ID_, that will be used in the slurm scripts
    - **SLURM-scripts in the _slurm_scripts_ folder pertaining to s8_4**:
        - `s8_4_LSTM_classification_functions.py`: containing functions for parameter search and training
        - `run_s8_4_lstm_param_search.slurm`: runs parameter search by running `s8_4_LSTM_classification_param_search.py`
        - `run_s8_4_lstm_training.slurm`: runs final model training using optimal hyperparameters selected after parameter search via `s8_4_LSTM_classification_training.py`
   - Runs parameter search
   - Runs final model training using optimal hyperparameters selected after parameter search
   - Plotting training loss or learning rates
   - **Plotting ROC-AUCs of test set patients is possible in notebook s9_5!**
    
- **s9_2: Extraction of LLM embeddings**
    -  Load LLM (**GPU necessary**!!)
    -  Load patient sentences created in **s7_4**, and place a final prompt in front of them asking the LLM to summarise the patient's condition, creating final input
    -  Loop through all experimental setups (_baseline_last_day_,_baseline_vars_, _all_days_) and time periods (baseline - month 6):
        -  Feed forward final input per patient, and extract last hidden layer of LLM, and average pool across the input token dimension to yield a patient embedding with dimensions equal to the hidden layer dimension of the LLM
        -  Attention pooling was not implemented
        -  GPU memory requirements: depending on the input sequence length, but BioMistral-7B (with precision float16) with the longest sequence input (~ 6800 tokens) reached peak memory consumption of ~ 45 GBs
    - **SLURM-scripts in the _slurm_scripts_ folder pertaining to s9_2**:
        -`run_s9_2_extact_LLM_embedding.slurm`:
            - runs LLM embedding extraction by running `s9_2_LLM_extract_embedding.py`
            - parameters for the SLURM-script are defined in the SLURM-script itself, no .tsv file generation in this case!
      
- **s9_3: Split patients into train-test splits &  Raw data - LR & XGBoost model training with SHAP analysis**
    - Create 25 stratified train-test split of patients, which will be used for raw and LLM-based models later
    - Create a .tsv file containing a dataframe with the parameter-combinations for the given tasks (paramater search, training, SHAP-value calculation) pairing them up with _SLURM_TASK_ID_, that will be used in the slurm scripts
    - **SLURM-scripts in the _slurm_scripts_ folder pertaining to s9_3**:
        - `s9_3_baseline_ML_models_functions.py`: containing general functions for parameter search, training, testing and SHAP-value calculation
        - `run_s9_3_raw_data_param_search.slurm`: runs parameter search by running `s9_3_baseline_ML_models_param_search.py`
        - `run_s9_3_raw_data_train.slurm`: runs final model training using optimal hyperparameters selected after parameter search via `s9_3_baseline_ML_models_training.py`
        - `run_s9_3_raw_data_testing.slurm`:by running `s9_3_baseline_ML_models_testing.py`, it extracts prediction probabilities of final trained models on the test set patients, as well as calculates ROC-AUC values using these probabilites and ground truth labels. **Uses a different .tsv file as input for _SLURM_TASK_ID_ pairing as all other s9_3 slurm scripts, for details check s9_3 notebook!**
        - `run_s9_3_raw_data_SHAP_calculation.slurm`: Runs SHAP-value calculation via `s9_3_baseline_ML_models_SHAP_calculation.py`, which also contains specific functions for SHAP-calculation
    - Runs parameter search (**slow, use SLURM-script**)
    - Runs final model training using optimal hyperparameters selected after parameter search (**slow, use SLURM-script**)
    - Create a .tsv file containing a dataframe with the parameter-combinations for model testing, pairing it up with _SLURM_TASK_ID_, that will be used in the slurm script `run_s9_3_raw_data_testing.slurm`
    - Run SHAP-value calculation (**slow, use SLURM-script**)
    - Concatenate SHAP-values & raw input data & prediction probabilities into dataframes ==> **REQUIREMENT FOR SHAP VALUE PLOTTING**!
    - Plot results of SHAP-value analysis (per timepoint beeswarm / clustermap / curves over timepoints)
 

- **s9_4: Embeddings - LR & XGBoost model training & Raw vs. Embedding-models: comparison over time; cluster based on prediction loss; Kaplan-Meier analysis**
    - Create a .tsv file containing a dataframe with the parameter-combinations for the given tasks (paramater search, training) pairing them up with _SLURM_TASK_ID_, that will be used in the slurm scripts
    - **SLURM-scripts in the _slurm_scripts_ folder pertaining to s9_4**:
        - `s9_4_ML_on_LLM_embeddings_functions.py`: containing general functions for parameter search and training
        - `run_s9_4_param_search.slurm`: runs parameter search by running `s9_4_ML_on_LLM_embeddings_param_search.py`
        - `run_s9_4_training.slurm`: runs final model training using optimal hyperparameters selected after parameter search via `s9_4_ML_on_LLM_embeddings_training.py`
    - Runs parameter search (**slow, use SLURM-script**)
    - Runs final model training using optimal hyperparameters selected after parameter search (**slow, use SLURM-script**)
    - Run SHAP-value calculation: only tractable for XGboost models, & SHAP for embeddings dimensions is pointless (see s9_6 for integrated gradients instead for model interpretation of embedding-based models), **but this step is necessary for the subsequent concatenation step!**
    - Concatenate SHAP-values & raw input data & prediction probabilities into dataframes ==> **REQUIREMENT FOR FURTHER STEPS**!
    - Match raw and embedding-based models' prediction logits as scatterplots
    - Cluster patients by their prediction loss across multiple timepoints
    - Compare prediction probabilities of test set patients for relapse prediction with those of patients with favourable end-of-therapy outcome but lost to follow-up (therefore dropped from relapse model training) --> Checking if there is a systematic difference between patients with completed and lost-to follow-up
    - Compare easy- and hard-to-treat groups across publications --> Imperial et al using baseline cavity & sputum smear vs. Chang using baseline lung disease grading & GeneXpert ==> **approximate GeneXpert with sputum smear, as GeneXpert is missing**
    - Run Kaplan-Meier analysis with Imperial et al.'s easy-/hard-to-treat (ETT/HTT) groups, and low/high relapse risk groupings derived from raw or embedding-based models (low/high: lowest/highest 30th percentile of prediction loss patients)
    - Perform a non-inferiority analysis across 4 and 6-month treatment arms for all groups derived in previous step (ETT/HTT, low/high risk)
    - Run a cost-effectiveness analysis (decision-curve & break even analysis)
 




