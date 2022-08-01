
###############################################
###   Train & Register XGBoost Core Model   ###
###############################################

###################
### Credentials ###
###################

import keyring
import runpy
import os
import urllib3
urllib3.disable_warnings()

### run script that contains username, password, hostname, working directory, and output directory
    ### ...OR define directly in this script
from password_poc import hostname, output_dir, wd
runpy.run_path(path_name='password_poc.py')
username = keyring.get_password('cas', 'username')
password = keyring.get_password('cas', username)
metadata_output_dir = 'outputs'

###################
### Environment ###
###################

import swat
import pandas as pd

port = 443
os.environ['CAS_CLIENT_SSL_CA_LIST']=str(wd)+str('/ca_cert_poc.pem')
conn =  swat.CAS(hostname, port, username=username, password=password, protocol='http')
print(conn)
print(conn.serverstatus())

#############################
### Identify Table in CAS ###
#############################

### caslib and table to use in modeling
caslib = 'Public'
in_mem_tbl = 'AML_BANK_PREP'

### load table in-memory if not already exists in-memory ###
if conn.table.tableExists(caslib=caslib, name=in_mem_tbl).exists<=0:
    conn.table.loadTable(caslib=caslib, path=str(in_mem_tbl+str('.sashdat')), 
                         casout={'name':in_mem_tbl, 'caslib':caslib, 'promote':True})

### show table to verify
conn.table.tableInfo(caslib=caslib, wildIgnore=False, name=in_mem_tbl)

########################
### Create Dataframe ###
########################

dm_inputdf =  conn.CASTable(in_mem_tbl, caslib=caslib).to_frame()

### read csv from defined 'data_dir' directory
#dm_inputdf = pd.read_csv(str(data_dir)+str('/')+in_mem_tbl+str('.csv'))

### print columns for review of model parameters
print(dm_inputdf.dtypes)

########################
### Model Parameters ###
########################

# import python libraries
import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import shuffle
from pathlib import Path

xgb_params = {
             'base_score': 0.5, 
             'booster': 'gbtree', 
             'eval_metric': 'auc', 
             'colsample_bytree': 1, 
             'colsample_bylevel': 1, 
             'colsample_bynode': 1, 
             'gamma': 0, 
             'grow_policy': 'depthwise', 
             'learning_rate': 0.1, 
             'max_bin': 256, 
             'max_delta_step': 0, 
             'max_depth': 3, 
             'max_leaves': 0, 
             'min_child_weight': 1, 
             'nthread': None, 
             'num_parallel_tree': 1, 
             'objective': 'binary:logistic', 
             'predictor': 'auto', 
             'process_type': 'default', 
             'refresh_leaf': 1, 
             'reg_alpha': 0, 
             'reg_lambda': 1, 
             'sampling_method': 'uniform', 
             'scale_pos_weight': 1, 
             'seed': None, 
             'seed_per_iteration': False, 
             'sketch_eps': 0.03, 
             'subsample': 1, 
             'tree_method': 'auto'
             } 
print(xgb_params)

### XGBOOST CORE
xgb_params_train = {'num_boost_round': 100,
                    'early_stopping_rounds': 50}
cutoff = 0.1

### model manager information
model_name = 'xgboost_python'
project_name = 'Risk Score'
description = 'XGBoost Python'
model_type = 'gradient_boost'
predict_syntax = 'predict_proba'

### define macro variables for model
dm_dec_target = 'ml_indicator'
dm_partitionvar = 'analytic_partition'
create_new_partition = 'no' # 'yes', 'no'
dm_key = 'account_id' 
dm_classtarget_level = ['0', '1']
dm_partition_validate_val, dm_partition_train_val, dm_partition_test_val = [0, 1, 2]
dm_partition_validate_perc, dm_partition_train_perc, dm_partition_test_perc = [0.3, 0.6, 0.1]

### create list of rejected predictor columns
rejected_predictors = [
    'atm_deposit_indicator', 
    'citizenship_country_risk', 
    'distance_to_bank',
    'distance_to_employer', 
    'income', 
    'num_acctbal_chgs_gt2000',
    'occupation_risk'
    ]

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
rejected_vars = rejected_predictors + macro_vars
for i in rejected_vars:
    dm_input.remove(i)

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

####################
### XGBOOST CORE ###
####################

### prediction parameter
predict_syntax = 'predict'

### convert data to matrices
xgb_train = xgb.DMatrix(X_train, y_train)
xgb_test = xgb.DMatrix(X_test, y_test)
xgb_valid = xgb.DMatrix(X_valid, y_valid)

### estimate & fit model
eval_list = [(xgb_valid, 'valid'), (xgb_test, 'test'), (xgb_train, 'train')]
dm_model = xgb.train(dtrain=xgb_train, params=xgb_params, evals=eval_list, **xgb_params_train)

### score full data
fullX = dm_inputdf.loc[:, dm_input]
fullX = xgb.DMatrix(fullX)
fully = dm_inputdf[dm_dec_target]
dm_scoreddf_prob = pd.DataFrame(dm_model.predict(fullX))
dm_scoreddf_prob.columns = [dm_predictionvar[1]]
dm_scoreddf_prob[dm_predictionvar[0]] = 1-dm_scoreddf_prob[dm_predictionvar[1]]
def binary_col (row):
    if row[dm_predictionvar[1]] > cutoff:
        return 1
    else:
        return 0
    
dm_scoreddf_prob[dm_classtarget_intovar] = dm_scoreddf_prob.apply (lambda row: binary_col(row), axis=1)
dm_scoreddf = dm_scoreddf_prob

### create tables with predicted values
trainProba = dm_model.predict(xgb_train)
testProba = dm_model.predict(xgb_test)
validProba = dm_model.predict(xgb_valid)
trainData = pd.concat([y_train.reset_index(drop=True), pd.Series(data=trainProba)], axis=1)
testData = pd.concat([y_test.reset_index(drop=True), pd.Series(data=testProba)], axis=1)
validData = pd.concat([y_valid.reset_index(drop=True), pd.Series(data=validProba)], axis=1)

### print model & results
cols = X_train.columns
predictors = np.array(cols)
tn, fp, fn, tp = confusion_matrix(fully, dm_scoreddf[dm_classtarget_intovar]).ravel()
print(description)
print(description)
print('model_parameters')
print(dm_model)
print(' ')
print('confusion_matrix:')
print('(tn, fp, fn, tp)')
print((tn, fp, fn, tp))
print('classification_report:')
print(classification_report(fully, dm_scoreddf[dm_classtarget_intovar]))

### print scoring columns
print(' ')
print('***** 5 rows from dm_scoreddf *****')
print(dm_scoreddf.head(5))
print(' ')
print('***** scoring columns *****')
print((', '.join(dm_input)))
print(dm_input)
print(*dm_input)

#######################################
### Register Model in Model Manager ###
## Ensure Model Does Not Exist in MM ##
##### Using PZMM Zips Up Metadata #####
#######################################

from sasctl import Session
import sasctl.pzmm as pzmm
import shutil
import json

### define macro vars for model manager
input_vars = X_train
scoring_targets = y_train
class_labels = ['EM_EVENTPROBABILITY', 'EM_CLASSIFICATION']
event_prob_var = class_labels[0]
target_event = dm_classtarget_level[1]
num_target_categories = len(dm_classtarget_level)
predict_method = str('{}.')+str(predict_syntax)+str('({})')
output_vars = pd.DataFrame(columns=class_labels, data=[[0.5, 'A']])

### create session in cas
sess=Session(hostname, username=username, password=password, verify_ssl=False, protocol="http")

### create directories for metadata
output_path = Path(output_dir) / metadata_output_dir / model_name
if output_path.exists() and output_path.is_dir():
    shutil.rmtree(output_path)

### create output path
os.makedirs(output_path)

### create python requirements file
requirements = [
    {
        "step":"import math, pickle, pandas as pd, numpy as np, settings",
        "command":"pip3 install math==3.10.5 pickle==3.10.5 numpy==1.20.3 pandas==1.3.4 settings==0.2.2"
    }
]
requirementsObj = json.dumps(requirements, indent = 4)
with open(str(output_path)+str('/requirements.json'), 'w') as outfile:
    outfile.write(requirementsObj)

### create metadata and import to model manager
pzmm.PickleModel.pickleTrainedModel(_, dm_model, model_name, output_path)
pzmm.JSONFiles().writeVarJSON(input_vars, isInput=True, jPath=output_path)
pzmm.JSONFiles().writeVarJSON(output_vars, isInput=False, jPath=output_path)
pzmm.JSONFiles().calculateFitStat(trainData=trainData, testData=testData, validateData=validData, jPath=output_path)
pzmm.JSONFiles().generateROCLiftStat(dm_dec_target, int(target_event), conn, trainData=trainData, testData=testData, validateData=validData, jPath=output_path)
pzmm.JSONFiles().writeFileMetadataJSON(model_name, jPath=output_path)
pzmm.JSONFiles().writeModelPropertiesJSON(
    modelName=model_name, 
    modelDesc=description,
    targetVariable=dm_dec_target,
    modelType=model_type,
    modelPredictors=predictors,
    targetEvent=target_event,
    numTargetCategories=num_target_categories,
    eventProbVar=event_prob_var,
    jPath=output_path,
    modeler=username)
pzmm.ImportModel().pzmmImportModel(output_path, model_name, project_name, input_vars, scoring_targets, predict_method, metrics=class_labels, force=True)

