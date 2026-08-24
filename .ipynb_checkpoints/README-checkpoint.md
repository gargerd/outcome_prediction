**This Readme describes the overall structure of the outcome_prediction repository**


# General overview

The repo can be split into multiple sections:

## Notebooks s0 to s4: preprocessing
- **s0**: notebook subsetting patients to individuals with available phase 3 outcome labels
- **s1**: subsetting preprocessed data from the repo preprocessing all TB-PACTS data (version 2021 August, https://github.com/gargerd/TBPACTS_preprocessing) to the data of the patients with available phase 3 outcome labels ==> requires the output of the TBPACTS_preprocessing repo!
- **s2-s4**: notebooks performing further data preprocessing by dataset type, and yielding one final dataframe per dataset type
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
    - cm: Concomitant medication taken
    - ce: Clinical events
    - su: Substance use
    - ae: Adverse events 

## Notebooks s5_1 to s5_3: concatenation of individual dataset type dataframes 
- **s5_1**: creating booelan dataframe indicating variable availability for patients, later used for variable and patient selection
- **s5_2**: for each patient, load their individual dataset type dataframe into nested dictionaries and save them as input to the next step
- **s5_3**: merge the dataframes from the nested dictionaries from s5_2 of the different dataset types into one "master" dataframe
     - Include dataset types, that were measured at some or all clinical visits during therapy period, and have a tractable number of individual variables (dm, mb, vs, re, lb, ms, ce, su)
     - Dataset types with values originating from one timepoint with thousands of individual variables (mh), or dataset types with daily values during therapy (dr_reg),  or daily values during therapy +  thousands of individual variables (cm, cmdos, ae) will be added at later timepoint
- **If any changes made in notebooks s0-s4, rerun s5_1 - s5_3 (in that order) for the changes to be incorporated in the downstream analysis (notebooks >s7)**
- 

## Notebook s6: outcome label extraction
- Extracts outcome data labels per patient for the studies TB-1018, TB-1020, TB-1021, TB-1022, TB-1030
- Returns multiple outcome labels: outcome at end-of-therapy, at 12/18 months after start-of-therapy, 24 months after end-of-therapy, etc..


## Notebooks s7_0 to s7_4: concatenation of individual dataset type dataframes 
