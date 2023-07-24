
##############################################
###     Train & Register MLFlow Model      ###
##############################################

###################
### Credentials ###
###################

import keyring
import runpy
import os
import urllib3
urllib3.disable_warnings()

### run script that contains username, password, hostname, working directory, and output directory
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

#dm_inputdf = pd.read_csv(str(data_dir)+str('/')+in_mem_tbl+str('.csv'))

### print columns for review of model parameters
print(dm_inputdf.dtypes)

########################
### Model Parameters ###
########################

### import python libraries
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, plot_roc_curve
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from pathlib import Path
import mlflow
from mlflow.models.signature import infer_signature

logit_params = {
             'penalty': 'l2', 
             'dual': False, 
             'tol': 0.0001, 
             'fit_intercept': True, 
             'intercept_scaling': 1, 
             'class_weight': None, 
             'random_state': None, 
             'solver': 'newton-cg', 
             'max_iter': 100, 
             'multi_class': 'auto', 
             'verbose': 0, 
             'warm_start': False, 
             'n_jobs': None, 
             'l1_ratio': None
             } 
print(logit_params)

### model manager information
model_name = 'logit_python_mlflow'
project_name = 'Risk Score'
description = 'Logistic Regression'
model_type = 'logistic_regression'
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
plt.hist(x=dm_scoreddf_prob[dm_predictionvar[1]], bins=100)

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

### mlflow run logs
if use_mlflow == 'yes':
    signature = infer_signature(X_train, dm_model.predict(X_test))
    mlflow.sklearn.log_model(dm_model, "model", signature=signature)
    mlflow.sklearn.eval_and_log_metrics(dm_model, X_test, y_test, prefix='p_')
    mlflow_id = mlflow.active_run().info.run_uuid
    mlflow.end_run()

#######################################
### Register Model in Model Manager ###
## Ensure Model Does Not Exist in MM ##
##### Using PZMM Zips Up Metadata #####
#######################################

from sasctl import Session
import sasctl.pzmm as pzmm
from sasctl._services.model_repository import ModelRepository as mr
import shutil

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

### create metadata and push to model manager
if use_mlflow == 'yes':
    mlflow_path = Path(str('./mlruns/')+str(mlflow_run_to_use)+str('/')+str(mlflow_id)+str('/artifacts/model'))
    mflow_dirs = ['artifacts', 'metrics', 'params', 'tags']
    for dirs in mflow_dirs:
        mlflow_path_artifacts = Path(str('./mlruns/')+str(mlflow_run_to_use)+str('/')+str(mlflow_id)+str('/')+str(dirs))
        for file in os.listdir(mlflow_path_artifacts):
            source = str(mlflow_path_artifacts)+str('/')+str(file)
            destination = str(output_path)+str('/')+str(file)
            if os.path.isfile(source):
                shutil.copy(source, destination)
    vars, input_vars, output_vars = pzmm.MLFlowModel.readMLmodelFile(_, mlflow_path)
    class_labels = mlflow_class_labels
    event_prob_var = mlflow_class_labels[0]
    scoring_targets = None
    predict_method = str('{}.')+str(mlflow_predict_syntax)+str('({})')
    pzmm.PickleModel.pickleTrainedModel(_, _, model_name, output_path, mlFlowDetails=vars)
else:
    input_vars = X_train
    scoring_targets = y_train
    output_vars = pd.DataFrame(columns=class_labels, data=[[0.5, 'A']])
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
