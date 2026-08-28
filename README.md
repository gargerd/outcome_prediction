**This Readme describes the overall structure of the outcome_prediction repository**


This description provides a general overview and structure of the notebooks and python files in this repo. Each notebook contains more detailed descriptions of its steps

# Notebooks and SLURM-scripts used for producing main analysis
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

## Notebooks s8_4 to s9_4: Train & test raw & embedding-based models; run SHAP-analysis for raw-data models; run Kaplan-Meier analysis 
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
        - `run_s9_3_raw_data_testing.slurm`:by running `s9_3_baseline_ML_models_testing.py`, it extracts prediction probabilities of final trained models on the test set patients, as well as calculates ROC-AUC values using these probabilites and ground truth labels. ROC-AUCs are calculated across studies, within studies and within arms. **Uses a different .tsv file as input for _SLURM_TASK_ID_ pairing as all other s9_3 slurm scripts, for details check s9_3 notebook's cell `Create dataframe with TEST combinations for parallel computing` or see below!**
        - `run_s9_3_raw_data_SHAP_calculation.slurm`: Runs SHAP-value calculation via `s9_3_baseline_ML_models_SHAP_calculation.py`, which also contains specific functions for SHAP-calculation
    - Model training:
        - Runs parameter search (**slow, use SLURM-script**)
        - Runs final model training using optimal hyperparameters selected after parameter search (**slow, use SLURM-script**)
    - Model testing:
        - Create a .tsv file containing a dataframe with the parameter-combinations for model testing, pairing it up with _SLURM_TASK_ID_, that will be used in the slurm script `run_s9_3_raw_data_testing.slurm`
        - Run model testing: it extracts prediction probabilities of final trained models on the test set patients, as well as calculates ROC-AUC values using these probabilites and ground truth labels (**slow, use SLURM-script**). ROC-AUCs are calculated across studies, within studies and within arms.
        - Plot ROC-AUCs over therapy timepoints, however not not all details of the figures are final (i.e. figure titles are rudimentary) --> **s9_5 notebook creates the same plots but with better visualisation!**
    - SHAP- value analysis: 
        - Run SHAP-value calculation (**slow, use SLURM-script**)
        - Concatenate SHAP-values & raw input data & prediction probabilities into dataframes ==> **REQUIREMENT FOR SHAP VALUE PLOTTING**!
        - **Important: observational SHAP-values were used, which consider correlation between inputs variables ==> put weight on all correlated variables, on those as well that are not being used by the model: https://arxiv.org/pdf/2006.16234**
        - Plot results of SHAP-value analysis (per timepoint beeswarm / clustermap / population median curves over timepoints, split by input value levels)
 

- **s9_4: Embeddings - LR & XGBoost model training & Raw vs. Embedding-models: comparison over time; cluster based on prediction loss; Kaplan-Meier analysis**
    - Create a .tsv file containing a dataframe with the parameter-combinations for the given tasks (paramater search, training) pairing them up with _SLURM_TASK_ID_, that will be used in the slurm scripts
    - **SLURM-scripts in the _slurm_scripts_ folder pertaining to s9_4**:
        - `s9_4_ML_on_LLM_embeddings_functions.py`: containing general functions for parameter search and training
        - `run_s9_4_param_search.slurm`: runs parameter search by running `s9_4_ML_on_LLM_embeddings_param_search.py`
        - `run_s9_4_training.slurm`: runs final model training using optimal hyperparameters selected after parameter search via `s9_4_ML_on_LLM_embeddings_training.py`
    - Model training:
        - Runs parameter search (**slow, use SLURM-script**)
        - Runs final model training using optimal hyperparameters selected after parameter search (**slow, use SLURM-script**)
    - Model testing:
        - Run model testing: it extracts prediction probabilities of final trained models on the test set patients, as well as calculates ROC-AUC values using these probabilites and ground truth labels. ROC-AUCs are calculated across studies, within studies and within arms.
        - Plot ROC-AUCs over therapy timepoints, however not not all details of the figures are final (i.e. figure titles are rudimentary) --> **s9_5 notebook creates the same plots but with better visualisation!**
    - Concatenate test set prediction probabilites:
        - Run SHAP-value calculation: only tractable for XGboost models, & SHAP for embeddings dimensions is pointless (see s9_6 for integrated gradients instead for model interpretation of embedding-based models), **but this step is necessary for the subsequent concatenation step!**
        - Concatenate SHAP-values & raw input data & prediction probabilities into dataframes ==> **REQUIREMENT FOR FURTHER STEPS**!
    - Match raw and embedding-based models' prediction logits as scatterplots
    - Cluster patients by their prediction loss across multiple timepoints
    - Compare prediction probabilities of test set patients for relapse prediction with those of patients with favourable end-of-therapy outcome but lost to follow-up (therefore dropped from relapse model training) --> Checking if there is a systematic difference between patients with completed and lost-to follow-up
    - Compare easy- and hard-to-treat groups across publications --> Imperial et al using baseline cavity & sputum smear vs. Chang using baseline lung disease grading & GeneXpert ==> **approximate GeneXpert with sputum smear, as GeneXpert is missing**
    - Subgroup analysis:
        - Run Kaplan-Meier analysis with Imperial et al.'s easy-/hard-to-treat (ETT/HTT) groups, and low/high relapse risk groupings derived from raw or embedding-based models (low/high: lowest/highest 30th percentile of prediction loss patients)
        - Perform a non-inferiority analysis across 4 and 6-month treatment arms for all groups derived in previous step (ETT/HTT, low/high risk)
    - Run a cost-effectiveness analysis (decision-curve & break even analysis)

## Notebook s9_5: Plot ROC-AUCs across timepoints 
- **s9_5: Plot ROC-AUCs across timepoints**
    - Concatenate ROC-AUC values of raw and embedding models
        - For each prediction label, concatenate ROC-AUC values of raw data and embedding-based model that were calculated in **s9_3** and **s9_4**, in order to make the next plotting steps easier
        - For experimental setups with leave-one-study-out approach (train on one or multiple studies, test on one left-out study), trainin-testing is done once, in contrast to experimental setups using all studies via a 25 train-test split approach. Perform bootstrapping to get confidence intervals
    - Plot ROC-AUCs of individual experimental setups as line and stripplots over time
        - Plot ROC-AUCs of LR & XGBoost models across studies / per study /  per arm
        - Plot ROC-AUCs of LSTM models across studies / per study /  per arm
    - Compare ROC-AUCs of the different data input approaches and models per one experimental setup in one barplot, and compare them statistically
        - Compare **raw data model performances only**: LR and XGBoost models of last-visit, last-visits concatenated data inputs; as well as LSTM models
        - Compare **embedding-based model performances only**: LR and XGBoost models of common variables - last visit, all variables - all visits, all variables - all visits data input approaches
        - Compare **across raw and embedding based data input approaches**: `final_setup_dict` controls which data setups will be plotted

## Notebook s9_6: Calculate Integrated Gradients attributions for embedding models 
- **s9_6: Calculate Integrated Gradients attributions for embedding models**
    - **Notes on why IG is used**:
        - Interpretability for embedding-based models doesn't work wih SHAP values, because SHAP values of embedding dimensions are not interpretable, as they cannot be traced back to individual clinical variables. In addition, SHAP calculation for LR models with 4096 dimensions is computationally intractable
        -  Integrated gradients can offer an alternative solution for this, however it needs a pipeline that is differentiable through all of its steps
        -  The embedding extraction steps are differentiable, but from the 2 models used (LR and XGBoost), only the former is differentiable, therefore only logistic regression models can be used for IG
        -  **A more detailed description of the whole approach can be found in s9_6's the markdown cell `RUN IG - SLURM SCRIPT AVAILABLE`!**
    -  Run integrated gradients, which results in saving the token level attributions to the final predictions (**slow, use SLURM-script**)
    - **SLURM-scripts in the _slurm_scripts_ folder pertaining to s9_6**:
        - `s9_6_LLM_integrated_gradients_LR_functions.py`: containing general functions for data loading and IG
        - `run_s9_6_integrated_gradients.slurm`:
            - runs Integrated Gradients by running `s9_6_LLM_integrated_gradients_LR.py`, which results in saving the token level attributions to the final predictions
            - parameters for the SLURM-script are defined in the SLURM-script itself, no .tsv file generation in this case!
            - **Running IG is very memory heavy ==> check `run_s9_6_integrated_gradients.slurm` for details and recommendations to avoid out-of-memory error (recommendations based on experience with A100 or H100 GPUs with 80GB RAM)**
    -  Match up tokens with their attributions, then group the tokens by variables, and sum the attributions of tokens belonging to one variable to yield one attribution score per variable
    -  Visualisation of attributions:
        - Clustering participants via clustermaps (x-axis: patients, y-axis: variables) based on their scaled IG attributions --> reliable implemented only for the common variable - last-visit data input approaches
        - Beeswarm-like scatterplots:
            - Calculate correlation between the attribution value and the raw input values of the variables --> this will be used later to select for the top 40 variables to plot in the case of `all visits` data input approach, as the number of variables is very large (timepoint of measurement is appended as a predix, i.e. Week_1_Haemoglobin)
            - Plot SHAP-style beeswarm plots of IG attributions per data input approach and per timepoint --> for `all visits` data input approach, select top 40 variables to plot for 3 different cases:
                - Top 40 variables with the largest absolute attributions
                - Top 40 variables with the largest absolute correlation between raw input values used for embedding and their attribution scores
                - Top 40 sparse variables with largest absolute attributions 
            - Save data for plotting Spearman correlations of selected common variable - last-visit approaches later (see details in cell `Save data for HH cluster - input data for Fig. 4's Spearman correlation plot`)
        - Plot top variables for both top-performing data input approaches (common vars. - last visit; all vars. - all visits):
            - common vars. - last visit:  population median curves of IG attributions, plotted over timepoints and split by input value levels
            - all vars. - all visits: plot beeswarm plots of variables with large absolute attributions & large correlation with input variables --> these are radiological finding (re) variables
    - Spearman-correlation between IG attributions of variables and their input values
        - Plot Spearman-correlation between IG attributions of variables and their input values for common vars. - last visit approaches of EOT outcome and relapse labels --> inputs for these plot come from **S9_3** notebook's `Save data for HH cluster - input data for Fig. 4's Spearman correlation plot` cell, and the cell of this notebook with the same title
        - Save correlations as a supplementary data. Correlation between variables attributions and their clinical values are calculated within arms, similarly to the beeswarm plots from before
    

# Notebooks and SLURM-scripts used for producing experimental analysis
 These notebooks & their respective SLURM-scripts contains experimental analysis in addition to the main analysis steps

## Notebook s9_7: Building ensemble models using prediction probabilities of raw and embedding-based model + metrics 
- **s9_7: Building ensemble models using prediction probabilities of raw and embedding-based model + metrics**
    - General approach:
        - During training (**s9_3** and **s9_4**), if `calibrate_model` is set to True, each trained model gets calibrated (isotonic method), and prediction probabilities of the uncalibrated and calibrated models are saved. In addition, some metrics using the uncalibrated and calibrated probabilities are also derived
        - In addition, using two distance metrics (Mahalanobis and KNN), the distance of the input data of the given patient is checked against the other patients, yielding information about how out-of-distribution the given patient is compared to the rest
        - Using these metrics (_see below_) of raw and embedding-based models of the same experimental setup as inputs (i.e. LR and XGboost models of _raw common-vars. - last-visit data_, and LR models of _all variables - all visit_ for the exp. setup `tb21_22_2984_pats_22_vars_relapse_without_dr_reg_ext_pats`), train an ensemble model predicting the same label --> for some patients the raw models are doing better, for some the embedding based one. This way we could leverage this phenomenon
    - The following confidence metrics are calculated:
        - Uncalibrated prediction probability
        - Calibrated prediction probability
        - Margin: pred. prob calibrated - 0.5
        - Entropy confidence: 1- entropy/log(2) --> entropy is calculated with calibrated probabilities
        - Delta calibration: pred. prob calibrated - pred. prob uncalibrated
        - Mahalanobis confidence value: Sigmoid(Mahal. distance to mean of training distribution)
        - KNN confidence value: Sigmoid((1/average distance) to its k nearest neighbours in the training data)
    
    - No SLURM-script as the parameter search and training are tractable (~2 hours)
    - **SUMMARY: No singificant improvement was seen with the ensemble model, therefore not included in main analysis**

## Notebook s9_8: LSTM on the last hidden layer of the LLM, without pooling
- **s9_8: LSTM on the last hidden layer of the LLM, without pooling**
    - General approach:
        - In all previous steps, after a forward pass of the input text, the last hidden layer's output (shaped as num_of_input_tokens x embedding_dimension) was mean pooled  performed across the input token dimension, yielding an embedding shaped as (1,embedding_dimension). Here, the idea is that we don't perform mean pooling, but train an LSTM on the full last hidden layer of the LLM (shaped as num_of_input_tokens x embedding_dimension), to extract the most important information from the input token dimension of the last hidden layer
        
    - Create a .tsv file containing a dataframe with the parameter-combinations for the given tasks (paramater search, training) pairing them up with _SLURM_TASK_ID_, that will be used in the slurm scripts
    - **SLURM-scripts in the _slurm_scripts_ folder pertaining to s9_8**:
        - `s9_8_LSTM_on_LLM_embeddings_functions.py`: containing general functions for LLM and LSTM
        - `run_s9_8_LSTM_on_LLM_embeddings.slurm`:
            - runs training by running `s9_8_LSTM_on_LLM_embeddings.py`
            - Check parameter settings for GPU settings!
    - Train LSTM on full LLM hidden layer --> Computationally also very heavy, training of one timepoint takes ~ 26 hours (**slow, use SLURM-script**)
    - Check training results
    - **SUMMARY: No singificant improvement was seen with the ensemble model, therefore not included in main analysis**

## Notebooks s9_9 to s9_12: Train & test raw & embedding-based SURVIVAL ANALYSIS models, within therapy durations (4 or 6-month arms); run SHAP-analysis for raw-data models;
- **s9_9: Raw data - SURVIVAL model training with SHAP analysis**
    - General approach:
        - Compared to the binary prediction of relapse in **s9_3** and **s9_4**, survival analysis is performed here
        - **Survival analysis is performed within therapy durations (within 4-month or 6-month arms)**
        - Using two timescales: relapse is counted either from start-of-therapy, or end-of-therapy, with no specific timepoint of right-censoring!
        - Models used: XGboost, CoxnetSurvival (~Cox PH model with ElasticNet penalty) and pycox's LogisticHazard (neural network, which discretises the follow-up period into a fixed set of time intervals, 6 in our case)
    - Structure of this notebook is really similar to that of an earlier version of **s9_3** --> **SLURM-script are analogously strucuted as those of s9_3**
        - Create a .tsv file containing a dataframe with the parameter-combinations for model testing, pairing it up with _SLURM_TASK_ID_, that will be used in the following slurm scripts
            -  **SLURM-scripts in the _slurm_scripts_ folder pertaining to s9_9**:
                - `s9_9_baseline_survival_models_functions.py`: containing general functions for parameter search, training, testing and SHAP-value calculation
                - `run_s9_9_raw_survival_param_search.slurm`: runs parameter search by running `s9_9_baseline_survival_models_param_search.py`
                - `run_s9_9_raw_survival_train.slurm`: runs final model training using optimal hyperparameters selected after parameter search via `s9_9_baseline_survival_models_training.py`
                - `run_s9_9_raw_survival_testing.slurm`:by running `s9_9_baseline_survival_models_testing.py`, it extracts prediction probabilities of final trained models on the test set. **Uses a different .tsv file as input for _SLURM_TASK_ID_ pairing as all other s9_9 slurm scripts, for details check s9_9 notebook's cell `Create dataframe with TEST combinations for parallel computing`!**
        - Parameter search, model training, testing, SHAP-values, (**all of them slow, use SLURM-scripts**) etc..        
    - **SUMMARY: Performance was better in the 4-month arms (0.6-0.65 C-index), as expected due to the higher number of relapses, and SHAP-analysis yielded similar SHAP values as those of s9_3. In the 6-month arms, the performance was worse (0.5-0.6) C-index all throughout the timepoints, which made the SHAP analysis not that reliable as well.** 


- **s9_10: Embedding-based - SURVIVAL model training**
    - Same general approach as  **s9_9** (survival models trained within 4 or 6 month arms, two time origins for relapse, 3 models)
    - Create a .tsv file containing a dataframe with the parameter-combinations for the given tasks (paramater search, training) pairing them up with _SLURM_TASK_ID_, that will be used in the slurm scripts
    - **SLURM-scripts in the _slurm_scripts_ folder pertaining to s9_10**:
        - `s9_10_survival_on_LLM_embeddings_functions.py`: containing general functions for parameter search and training
        - `run_s9_10_llm_survival_param_search.slurm`: runs parameter search by running `s9_10_survival_on_LLM_embeddings_param_search.py`
        - `run_s9_10_llm_survival_train.slurm`: runs final model training using optimal hyperparameters selected after parameter search via `s9_10_survival_on_LLM_embeddings_training.py`
    - Parameter search (**slow, use SLURM-scripts**)
    - Model training (**slow, use SLURM-scripts**)
    - Model testing



- **s9_11: Plot C-index across timepoints**
  - Concatenate C-index values of raw and embedding models
        - For each prediction label, concatenate C-index values of raw data and embedding-based model that were calculated in **s9_9** and **s9_10**, in order to make the next plotting steps easier
    - Plot C-index values of individual experimental setups as line and stripplots over time
        - Plot C-index values of CoxnetSurvival & XGBoost models across studies / per study /  per arm
    - Compare C-index values of the different data input approaches and models per one experimental setup in one barplot, and compare them statistically
        - Compare **raw data model performances only**: CoxnetSurvival & XGBoost models of last-visit, last-visits concatenated data inputs; as well as LSTM models
        - Compare **embedding-based model performances only**: CoxnetSurvival & XGBoost models of common variables - last visit, all variables - all visits, all variables - all visits data input approaches
        - Compare **across raw and embedding based data input approaches**: `final_setup_dict` controls which data setups will be plotted
    - **SUMMARY: Performance was similar in the 4-month arms to those of the raw model with all data input approaches (0.6-0.65 C-index), as expected due to the higher number of relapses. For the 6-month arms, the _common variables / all variables - all visits_ approaches seemed to perform better compared to the raw models, both in the across-study and within arm C-index values.** 


- **s9_12: Calculate Integrated Gradients attributions for embedding survival models**
    - Analog of **s9_6**, see details there
    - Differences: CoxnetSurvival is used here instead of LR & that the models were trained within therapy duration (4 or 6-moth arms pooled)
      
    - **SLURM-scripts in the _slurm_scripts_ folder pertaining to s9_12**:
        - `s9_12_LLM_survival_integrated_gradients_functions.py`: containing general functions for data loading and IG
        - `run_s9_12_LLM_survival_integrated_gradients.slurm`:
            - runs Integrated Gradients by running `s9_12_LLM_survival_integrated_gradients.py`, which results in saving the token level attributions to the final predictions
            - parameters for the SLURM-script are defined in the SLURM-script itself, no .tsv file generation in this case!
            - **Running IG is very memory heavy ==> check `run_s9_12_LLM_survival_integrated_gradients.slurm` for details and recommendations to avoid out-of-memory error (recommendations based on experience with A100 or H100 GPUs with 80GB RAM)**
    - **SUMMARY: really similar results to those of s9_6**



