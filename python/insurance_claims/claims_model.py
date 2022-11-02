#########################################################
###      Train & Register Insurance Claims Model      ###
#########################################################

###################
### Credentials ###
###################

import getpass
import runpy
import os
import urllib3
urllib3.disable_warnings()

username = getpass.getpass("Username: ")
password = getpass.getpass("Password: ")
output_dir = os.getcwd()
metadata_output_dir = 'outputs'

###################
### Environment ###
###################

import swat
import pandas as pd

hostname = 'https://fsbulab.unx.sas.com/cas-shared-default-http'
port = 443
conn = swat.CAS(hostname, port, username, password, protocol="https")
print(conn)
print(conn.serverstatus())

#############################
### Identify Table in CAS ###
#############################

### caslib and table to use in modeling
caslib = 'Public'
in_mem_tbl = 'PURE_PREMIUM_W_TRANSFORMS'

### load table in-memory if not already exists in-memory
if conn.table.tableExists(caslib=caslib, name=in_mem_tbl).exists<=0:
    conn.table.loadTable(caslib=caslib, path=str(in_mem_tbl+str('.sashdat')), 
                         casout={'name':in_mem_tbl, 'caslib':caslib, 'promote':True})

### show table to verify
conn.table.tableInfo(caslib=caslib, wildIgnore=False, name=in_mem_tbl)

########################
### Create Dataframe ###
########################

#dm_inputdf =  conn.CASTable(in_mem_tbl, caslib=caslib).to_frame()

### read csv from defined 'data_dir' directory
data_dir = 'C:/Users/chparr/OneDrive - SAS/pure_premium'
dm_inputdf = pd.read_csv(str(data_dir)+str('/')+in_mem_tbl+str('.csv'))

### print columns for review of model parameters
print(dm_inputdf.dtypes)

### import python libraries
import numpy as np
from sklearn.linear_model import TweedieRegressor
from sklearn.metrics import mean_tweedie_deviance, mean_absolute_error, mean_squared_error
from sklearn.utils import shuffle
from pathlib import Path

tweedie_params = {
             'power': 1.5, 
             'alpha': 0.1, 
             'maxiter': 10000
             } 
print(tweedie_params)

### model manager information
model_name = 'tweedie_python'
project_name = 'Pure Premium'
description = 'Tweedie GLM'
model_type = 'GLM'
predict_syntax = 'predict_proba'

### define macro variables for model
dm_dec_target = 'PurePremium'
dm_partitionvar = 'PartInd'
create_new_partition = 'no' # 'yes', 'no'
dm_key = 'uniqueRecordID' 
#dm_classtarget_level = ['0', '1']
dm_partition_validate_val, dm_partition_train_val, dm_partition_test_val = [0, 1, 2]
dm_partition_validate_perc, dm_partition_train_perc, dm_partition_test_perc = [0.3, 0.6, 0.1]

### create list of rejected predictor columns
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
rejected_predictors = [i for i in dm_inputdf if i not in keep_predictors]

### mlflow
use_mlflow = 'yes' # 'yes', 'no'
mlflow_run_to_use = 0
mlflow_class_labels =['TENSOR']
mlflow_predict_syntax = 'predict'

### var to consider in bias assessment
bias_var = 'cross_border_trx_indicator'

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

#####################
### Training Code ###
#####################

### estimate & fit model
dm_model = LogisticRegression(**logit_params)
dm_model.fit(X_train, y_train)

### score full data
fullX = dm_inputdf.loc[:, dm_input]
fully = dm_inputdf[dm_dec_target]
plot_roc_curve(dm_model, fullX, fully)
dm_scoreddf_prob = pd.DataFrame(dm_model.predict_proba(fullX), columns=dm_predictionvar)
dm_scoreddf_class = pd.DataFrame(dm_model.predict(fullX), columns=[dm_classtarget_intovar])
dm_scoreddf = pd.concat([dm_scoreddf_prob, dm_scoreddf_class], axis=1)

### create tables with predicted values
trainProba = dm_model.predict_proba(X_train)
testProba = dm_model.predict_proba(X_test)
validProba = dm_model.predict_proba(X_valid)
trainData = pd.concat([y_train.reset_index(drop=True), pd.Series(data=trainProba[:,1])], axis=1)
testData = pd.concat([y_test.reset_index(drop=True), pd.Series(data=testProba[:,1])], axis=1)
validData = pd.concat([y_valid.reset_index(drop=True), pd.Series(data=validProba[:,1])], axis=1)

### print model & results
predictions = dm_model.predict(X_test)
cols = X_train.columns
predictors = np.array(cols)
tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
print(description)
print(description)
print('model_parameters')
print(dm_model)
print(' ')
print('model_performance')
print('score_full:', dm_model.score(fullX, fully))
print('score_train:', dm_model.score(X_train, y_train))
print('score_test:', dm_model.score(X_test, y_test))
print('score_valid:', dm_model.score(X_valid, y_valid))
print('confusion_matrix:')
print('(tn, fp, fn, tp)')
print((tn, fp, fn, tp))
print('classification_report:')
print(classification_report(y_test, predictions))

### print logit odds ratios
orat = np.exp(dm_model.coef_, out=None)
c1 = np.vstack([predictors,orat])
c2 = np.transpose(c1)
c = pd.DataFrame(c2, columns=['predictors', 'odds_ratio'])
print('intercept:')
print(dm_model.intercept_)
print('odds_ratios:')
print(c)

#######################################
### Register Model in Model Manager ###
## Ensure Model Does Not Exist in MM ##
##### Using PZMM Zips Up Metadata #####
#######################################

from sasctl import Session
import sasctl.pzmm as pzmm
from sasctl.services import model_repository as modelRepo 
from sasctl.tasks import register_model
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
# pzmm.ImportModel().pzmmImportModel(output_path, model_name, project_name, input_vars, scoring_targets, predict_method, metrics=class_labels, force=True)

# alternative model registration
pzmm.ScoreCode().writeScoreCode(input_vars, scoring_targets, model_name, predict_method, model_name + '.pickle', pyPath=output_path)
zip_file = pzmm.ZipModel.zipFiles(fileDir=output_path, modelPrefix=model_name, isViya4=True)
with sess:
    try:
        modelRepo.import_model_from_zip(model_name, project_name, zip_file, version='latest')
    except ValueError:
        modelRepo.create_project(project_name, caslib)
        modelRepo.import_model_from_zip(model_name, project_name, zip_file, version='latest')