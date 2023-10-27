
#####################################################
###      Train & Register SAS gbTree Model        ###
#####################################################

###################
### Credentials ###
###################

import keyring
import os
from pathlib import Path
import urllib3
import runpy
urllib3.disable_warnings()

### run script that contains username, password, hostname, working directory, and output directory
    ### ...OR define directly in this script
from password import hostname_cas, hostname_http, port_cas, port_http, protocol_cas, protocol_http, wd, output_dir, hostname_dev, port_dev, protocol_dev, cert_dir, token_sse, token_sse_refresh, token_sse_pem, hostname_sse, session_sse

runpy.run_path(path_name='password.py')
username = keyring.get_password('cas', 'username')
password = keyring.get_password('cas', username)
metadata_output_dir = 'outputs'

###################
### Environment ###
###################

import swat
import pandas as pd
from casauth import CASAuth

conn =  swat.CAS(hostname=hostname_cas, port=port_cas, username=username, password=password, protocol=protocol_cas)
### ssemonthly connection ###
#access_token = open(token_sse, "r").read()
#conn =  swat.CAS(hostname=hostname_sse, username=None, password=access_token, ssl_ca_list=token_sse_pem, protocol=protocol_http)
#conn = CASAuth(cert_dir, ssl_ca_list=token_sse_pem)
print(conn.serverstatus())

#############################
### Identify Table in CAS ###
#############################

### caslib and table to use in modeling
caslib = 'Public'
in_mem_tbl = 'FINANCIAL_SERVICES_PREP'

### load table in-memory if not already exists in-memory
if conn.table.tableExists(caslib=caslib, name=in_mem_tbl).exists<=0:
    conn.table.loadTable(caslib=caslib, path=str(in_mem_tbl+str('.sashdat')), 
                         casout={'name':in_mem_tbl, 'caslib':caslib, 'promote':True})
    
### show table to verify
conn.table.tableInfo(caslib=caslib, wildIgnore=False, name=in_mem_tbl)

### create names of tables for action set
astore_tbl = str(in_mem_tbl+str('_astore'))
cas_score_tbl = str(in_mem_tbl+str('_score'))
cas_out_tbl = str(in_mem_tbl+str('_model'))

########################
### Create Dataframe ###
########################

dm_inputdf = conn.CASTable(in_mem_tbl, caslib=caslib)

### print columns for review of model parameters
conn.table.columnInfo(table={"caslib":caslib, "name":in_mem_tbl})

########################
### Model Parameters ###
########################

### import packages
conn.loadactionset('decisionTree')
conn.loadactionset('astore')
conn.loadactionset('explainModel')
conn.loadactionset('fairAITools')
conn.loadactionset('percentile')
conn.loadactionset('modelPublishing')

### model arugments
xgb_params = dict(
    m=20,
    seed=12345,
    nTree=100,
    learningRate=0.1,
    subSampleRate=0.5,
    lasso=0,
    ridge=1,
    distribution="binary",
    maxBranch=2,
    maxLevel=5,
    leafSize=5,
    missing="useinsearch",
    minUseInSearch=1,
    nBins=50,
    quantileBin=True
    )

early_stop_params = dict(
    metric="MCR",
    stagnation=5,
    tolerance=0,
    minimum=False,
    threshold=0,
    thresholdIter=0
    )
print(xgb_params)
print(early_stop_params)

### model manager information
model_name = 'gbtree_action_python_finsvcs'
project_name = 'Financial Services'
description = 'gbtree_action'
model_type = 'gradient_boosting'

### define macro variables for model
dm_dec_target = 'event_indicator'
dm_partitionvar = 'analytic_partition' 
dm_key = 'account_id' 
dm_classtarget_level = ['0', '1']
dm_partition_validate_val, dm_partition_train_val, dm_partition_test_val = [0, 1, 2]

### var to consider in bias assessment
bias_var = 'gender'

### var to consider in partial dependency
pd_var1 = 'credit_score'
pd_var2 = 'net_worth'

##############################
### Final Modeling Columns ###
##############################

### create list of model variables
dm_input = list(dm_inputdf.columns.values)
macro_vars = (dm_dec_target + ' ' + dm_partitionvar + ' ' + dm_key).split()
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
#keep_predictors = [i for i in dm_input if i not in macro_vars]
#rejected_predictors = []
rejected_predictors = [i for i in dm_input if i not in keep_predictors]
rejected_vars = rejected_predictors # + macro_vars
for i in rejected_vars:
    dm_input.remove(i)

### create prediction variables
dm_predictionvar = [str('P_') + dm_dec_target + dm_classtarget_level[0], str('P_') + dm_dec_target + dm_classtarget_level[1]]
dm_classtarget_intovar = str('I_') + dm_dec_target

### create partition objects
train_part = str(dm_partitionvar)+str('=')+str(dm_partition_train_val)
test_part = str(dm_partitionvar)+str('=')+str(dm_partition_test_val)
valid_part = str(dm_partitionvar)+str('=')+str(dm_partition_validate_val)

#####################
### Training Code ###
#####################

dm_model = conn.decisionTree.gbtreeTrain(**xgb_params,
    earlyStop=early_stop_params,
    table=dict(caslib=caslib, name=in_mem_tbl, where=train_part),
    target=dm_dec_target,
    nominal=dm_dec_target,
    inputs=dm_input,
    encodeName=True,
    casOut=dict(caslib=caslib, name=cas_out_tbl, replace=True),
    saveState=dict(caslib=caslib, name=astore_tbl, replace=True)
    )

####################
###  Score Data  ###
####################

conn.decisionTree.dtreeScore(
    modelTable=dict(caslib=caslib, name=cas_out_tbl),
    table=dict(caslib=caslib, name=in_mem_tbl), 
    copyvars=[dm_dec_target, dm_partitionvar],
    casout=dict(caslib=caslib, name=cas_score_tbl, replace=True),
    encodeName=True,
    assessOneRow=True
    )

####################
### Scoring Code ###
####################

conn.decisionTree.gbtreeCode(
  modelTable=dict(caslib=caslib, name=cas_out_tbl),
  code=dict(casOut=dict(caslib=caslib, name=(str(description)+str('_scorecode')), replace=True, promote=False))
  )

##########################
### Partial Dependency ###
##########################

conn.explainModel.partialDependence (
        table=dm_inputdf,
        seed=12345,
        modelTable=dict(caslib=caslib, name=astore_tbl),
        predictedTarget=dm_predictionvar[1],
        analysisVariable=pd_var2,
        inputs=dm_input,
        output=dict(casOut=dict(caslib=caslib, name='partial_dependency', replace=True))
        )

###################
### SHAP Values ###
###################

import numpy as np
conn.loadactionset('transpose')

sample_size = 500
rand_obs = np.array(dm_inputdf[dm_key].sample(n=sample_size, random_state=12345).as_matrix())

for i in range(sample_size):
    obs_num = rand_obs[i].astype(int).item()
    obs_num = 136
    shapley_temp = dm_inputdf[dm_inputdf[dm_key] == obs_num]
    
    conn.explainModel.shapleyExplainer(
        table=dm_inputdf,
        query=shapley_temp,
        modelTable=dict(caslib=caslib, name=astore_tbl),
        modelTableType="astore",
        predictedTarget=dm_predictionvar[1],
        inputs=dm_input,
        depth=1,
        outputTables=dict(names=dict(ShapleyValues=dict(name='shapleyvalues', caslib=caslib, replace=True)))
        )
    
    conn.transpose.transpose(
        table=dict(caslib=caslib, name="shapleyvalues"),
        id="variable",
        casOut=dict(caslib=caslib, name="shapley_transpose", replace=True)
        )
    
    if i == 0:
        conn.table.copyTable(
            table=dict(caslib=caslib, name="shapley_transpose"),
            casOut=dict(caslib=caslib, name='shapley_rows', replace=True)
            )
        conn.table.copyTable(
            table=dict(caslib=caslib, name="shapleyvalues"),
            casOut=dict(caslib=caslib, name='shapley_cols', replace=True))
    else:
        conn.table.append(
            source=dict(caslib=caslib, name='shapley_transpose'),
            target=dict(caslib=caslib, name='shapley_rows')
            )
        conn.table.append(
            source=dict(caslib=caslib, name='shapleyvalues'),
            target=dict(caslib=caslib, name='shapley_cols')
            )

shapley_rows = conn.CASTable(caslib=caslib, name='shapley_rows').to_frame()
shapley_cols = conn.CASTable(caslib=caslib, name='shapley_cols').to_frame()

### may need to unload global scope tables first and save as perm tables
conn.table.promote(caslib=caslib, name='shapley_rows')
conn.table.promote(caslib=caslib, name='shapley_cols')

###################
### Assess Bias ###
###################

conn.fairAITools.assessBias(
        table = dict(caslib=caslib, name=in_mem_tbl),
        modelTable = dict(caslib=caslib, name=astore_tbl),
        modelTableType = "ASTORE",
        response = dm_dec_target,
        predictedVariables = dm_predictionvar,
        responseLevels = dm_classtarget_level,
        sensitiveVariable = bias_var
        )

######################################################
### Create Pandas Dataframes with Predicted Values ###
######################################################

import pandas as pd

score_astore = conn.CASTable(caslib=caslib, name=cas_score_tbl)
dm_scoreddf = conn.CASTable(caslib=caslib, name=score_astore).to_frame()
dm_scoreddf[dm_dec_target] = dm_scoreddf[dm_dec_target].astype(int)
trainData = dm_scoreddf[dm_scoreddf[dm_partitionvar]==dm_partition_train_val][[dm_dec_target, dm_predictionvar[1]]].rename(columns=lambda x:'0')
testData = dm_scoreddf[dm_scoreddf[dm_partitionvar]==dm_partition_test_val][[dm_dec_target, dm_predictionvar[1]]].rename(columns=lambda x:'0')
validData = dm_scoreddf[dm_scoreddf[dm_partitionvar]==dm_partition_validate_val][[dm_dec_target, dm_predictionvar[1]]].rename(columns=lambda x:'0')
trainData = pd.DataFrame(trainData)
testData = pd.DataFrame(testData)
validData = pd.DataFrame(validData)

### print model & results
print(dm_model)
conn.table.tableInfo(caslib=caslib, wildIgnore=False, name=astore_tbl)
print(conn.astore.describe(rstore=dict(name=astore_tbl, caslib=caslib), epcode=True).Description)
print(conn.astore.describe(rstore=dict(name=astore_tbl, caslib=caslib), epcode=True).InputVariables)
print(conn.astore.describe(rstore=dict(name=astore_tbl, caslib=caslib), epcode=True).OutputVariables)
print(conn.astore.describe(rstore=dict(name=astore_tbl, caslib=caslib), epcode=True).epcode)
model_astore = conn.CASTable(astore_tbl, caslib=caslib)

#########################################
###  Register Model in Model Manager  ###
#########################################

from pathlib import Path
from sasctl import Session
from sasctl import pzmm as pzmm
from sasctl.tasks import register_model, publish_model
from sasctl._services.model_repository import ModelRepository as mr
import shutil

### define macro vars for model manager
target_event = dm_classtarget_level[1]

### create session in cas
sess=Session(hostname, username=username, password=password, verify_ssl=False, protocol="http")

### create directories for metadata
output_path = Path(output_dir) / metadata_output_dir / model_name
if output_path.exists() and output_path.is_dir():
    shutil.rmtree(output_path)

### create output path
os.makedirs(output_path)

### create metadata and import to model manager
pzmm.JSONFiles().calculateFitStat(trainData=trainData, testData=testData, validateData=validData, jPath=output_path)
pzmm.JSONFiles().generateROCLiftStat(dm_dec_target, int(target_event), conn, trainData=trainData, testData=testData, validateData=validData, jPath=output_path)
file_list = os.listdir(output_path)
files = []
for i in file_list:
    new_dict = {'name':i, 'file':open(output_path / i), 'role':'Properties and Metadata'}
    files.append(new_dict)
with sess:
    reg_model = register_model(model_astore, model_name, project_name, files=files, force=True, version='latest')
    for file in files:
        mr.add_model_content(model_name, **file)
#   pub_model = publish_model(model_name, 'maslocal')
#   score_example = pub_model.score(input1=1, input2=2, etc.)
