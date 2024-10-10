###########################################################
###  Train Scikit Logit Model & SAS Viya ML Logit Model ###
###########################################################

########################
### Create Dataframe ###
########################

import pandas as pd

data_dir = "/workspaces/chris_parrish/sas_viya/data/financial_services"
workspace_dir = "/workspaces/chris_parrish/_chris_demo"
data_table = "financial_services_prep.csv"

dm_inputdf = pd.read_csv(Path(data_dir) / data_table, header=0)
print(dm_inputdf.dtypes)

########################
### Model Parameters ###
########################

### import python libraries
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import shuffle

### model manager information
model_name = 'logit_python_finsvcs'
project_name = 'Financial Services'
description = 'Logistic Regression'
model_type = 'logistic_regression'
predict_syntax = 'predict_proba'

### define macro variables for model
dm_dec_target = 'event_indicator'
dm_partitionvar = 'analytic_partition'
create_new_partition = 'no' # 'yes', 'no'
dm_key = 'account_id' 
dm_classtarget_level = ['0', '1']
dm_partition_validate_val, dm_partition_train_val, dm_partition_test_val = [0, 1, 2]
dm_partition_validate_perc, dm_partition_train_perc, dm_partition_test_perc = [0.3, 0.6, 0.1]

### create list of regressors
keep_predictors = [
    'net_worth',
    'credit_score',
    'num_dependents',
    'at_current_job_1_year',
    'credit_history_mos',
    'job_in_education',
    'num_transactions',
    'debt_to_income',
    'amount',
    'gender',
    'age',
    'job_in_hospitality'
    ]
#rejected_predictors = []

### mlflow
use_mlflow = 'no' # 'yes', 'no'
mlflow_run_to_use = 0
mlflow_class_labels =['TENSOR']
mlflow_predict_syntax = 'predict'

### var to consider in bias assessment
bias_var = 'gender'

### var to consider in partial dependency
pd_var1 = 'credit_score'
pd_var2 = 'net_worth'

### create partition column, if not already in dataset
if create_new_partition == 'yes':
    dm_inputdf = shuffle(dm_inputdf)
    dm_inputdf.reset_index(inplace=True, drop=True)
    validate_rows = round(len(dm_inputdf)*dm_partition_validate_perc)
    train_rows = round(len(dm_inputdf)*dm_partition_train_perc) + validate_rows
    test_rows = len(dm_inputdf)-train_rows
    dm_inputdf.loc[0:validate_rows,dm_partitionvar] = dm_partition_validate_val
    dm_inputdf.loc[validate_rows:train_rows,dm_partitionvar] = dm_partition_train_val
    dm_inputdf.loc[train_rows:,dm_partitionvar] = dm_partition_test_val

##############################
### Final Modeling Columns ###
##############################

### create list of model variables
dm_input = list(dm_inputdf.columns.values)
macro_vars = (dm_dec_target + ' ' + dm_partitionvar + ' ' + dm_key).split()
rejected_predictors = [i for i in dm_input if i not in keep_predictors]
rejected_vars = rejected_predictors # + macro_vars (include macro_vars if rejected_predictors are explicitly listed - not contra keep_predictors)
for i in rejected_vars:
    dm_input.remove(i)
print(dm_input)

### create prediction variables
dm_predictionvar = [str('P_') + dm_dec_target + dm_classtarget_level[0], str('P_') + dm_dec_target + dm_classtarget_level[1]]
dm_classtarget_intovar = str('I_') + dm_dec_target

##################
### Data Split ###
##################

### create train, test, validate datasets using existing partition column
dm_traindf = dm_inputdf[dm_inputdf[dm_partitionvar] == dm_partition_train_val]
X_train = dm_traindf.loc[:, dm_input]
y_train = dm_traindf[dm_dec_target]
dm_testdf = dm_inputdf.loc[(dm_inputdf[dm_partitionvar] == dm_partition_test_val)]
X_test = dm_testdf.loc[:, dm_input]
y_test = dm_testdf[dm_dec_target]
dm_validdf = dm_inputdf.loc[(dm_inputdf[dm_partitionvar] == dm_partition_validate_val)]
X_valid = dm_validdf.loc[:, dm_input]
y_valid = dm_validdf[dm_dec_target]

#####################
### Training Code ###
#####################

from sklearn.linear_model import LogisticRegression

### estimate & fit model
dm_model = LogisticRegression(
        tol=1e-8,
        fit_intercept=True,
        solver='newton-cg',
        verbose=0,
        max_iter=None
    )
dm_model.fit(X_train, y_train)

print('score_train:', dm_model.score(X_train, y_train))
print('score_test:', dm_model.score(X_test, y_test))
print('score_valid:', dm_model.score(X_valid, y_valid))

####################
### Pickle Model ###
####################

import pickle

pickle_file = 'financial_servies_pickle.pkl'
dm_pklpath = Path(workspace_dir)/pickle_file

with open(dm_pklpath, 'wb') as f:
	pickle.dump(dm_model, f)

###############
### SAS API ###
###############

from sasviya.ml.linear_model import LogisticRegression

### estimate & fit model
dm_model = LogisticRegression(
        tol=1e-8,
        fit_intercept=True,
        solver="newrap",
        selection=None,
        verbose=0,
        max_iter=None,
        max_time=None
        )
dm_model.fit(X_train, y_train)

print('score_train:', dm_model.score(X_train, y_train))
print('score_test:', dm_model.score(X_test, y_test))
print('score_valid:', dm_model.score(X_valid, y_valid))

#################################
### Save Model as Astore File ###
#################################

astore = "financial_services_astore"
dm_model.export(file=Path(workspace_dir)/astore, replace=True)

###########################
### Save Model in Table ###
###########################

model_table = "financial_services_model"
dm_model.save(Path(workspace_dir)/model_table)