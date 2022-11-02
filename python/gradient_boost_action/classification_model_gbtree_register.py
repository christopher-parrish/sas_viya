
#####################################################
###       Train & Publish SAS gbTree Model        ###
#####################################################

###################
### Credentials ###
###################

import keyring
import runpy
import os
from pathlib import Path
import urllib3
urllib3.disable_warnings()

### run script that contains username, password, hostname, working directory, and output directory
    ### ...OR define directly in this script
from password_poc import hostname, wd, output_dir
runpy.run_path(path_name='password.py')
username = keyring.get_password('cas', 'username')
password = keyring.get_password('cas', username)
metadata_output_dir = 'outputs'

###################
### Environment ###
###################

import swat

port = 443
os.environ['CAS_CLIENT_SSL_CA_LIST']=str(wd)+str('/ca_cert.pem')
conn =  swat.CAS(hostname, port, username=username, password=password, protocol='http')
print(conn.serverstatus())

#############################
### Identify Table in CAS ###
#############################

### caslib and table to use in modeling
caslib = 'Public'
in_mem_tbl = 'AML_BANK_PREP'

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

dm_inputdf =  conn.CASTable(in_mem_tbl, caslib=caslib)

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
model_name = 'gbtree_python'
project_name = 'Risk Score'
description = 'gbtree_python'
model_type = 'gradient_boost'

### define macro variables for model
dm_dec_target = 'ml_indicator'
dm_partitionvar = 'analytic_partition' 
dm_key = 'account_id' 
dm_classtarget_level = ['0', '1']
dm_partition_validate_val, dm_partition_train_val, dm_partition_test_val = [0, 1, 2]

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

### var to consider in bias assessment
bias_var = 'cross_border_trx_indicator'

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

### create partition objects
train_part = str(dm_partitionvar)+str('=')+str(dm_partition_train_val)
test_part = str(dm_partitionvar)+str('=')+str(dm_partition_test_val)
valid_part = str(dm_partitionvar)+str('=')+str(dm_partition_validate_val)

#####################
### Training Code ###
#####################

### estimate & fit model
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

### score full data
conn.decisionTree.dtreeScore(
    modelTable=dict(caslib=caslib, name=cas_out_tbl),
    table=dict(caslib=caslib, name=in_mem_tbl), 
    copyvars=[dm_dec_target, dm_partitionvar],
    casout=dict(caslib=caslib, name=cas_score_tbl, replace=True),
    encodeName=True,
    assessOneRow=True
    )

### create score code
conn.decisionTree.gbtreeCode(
  modelTable=dict(caslib=caslib, name=cas_out_tbl),
  code=dict(casOut=dict(caslib=caslib, name='gbtree_scorecode', replace=True, promote=False))
  )

####################
### Assess Model ###
####################

conn.percentile.assess(
  table=dict(caslib=caslib, name=cas_score_tbl),
  event="1",
  response=dm_dec_target,
  inputs=dm_predictionvar[1],
  cutStep=0.0001,
  casOut=dict(caslib=caslib, name='gbtree_python_assess', replace=True)
  )

### print model & results
print(dm_model)
conn.table.tableInfo(caslib=caslib, wildIgnore=False, name=astore_tbl)
print(conn.astore.describe(rstore=dict(name=astore_tbl, caslib=caslib), epcode=True).Description)
print(conn.astore.describe(rstore=dict(name=astore_tbl, caslib=caslib), epcode=True).InputVariables)
print(conn.astore.describe(rstore=dict(name=astore_tbl, caslib=caslib), epcode=True).OutputVariables)
print(conn.astore.describe(rstore=dict(name=astore_tbl, caslib=caslib), epcode=True).epcode)
model_astore = conn.CASTable(astore_tbl, caslib=caslib)

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

score_astore = conn.CASTable(cas_score_tbl)
dm_scoreddf = conn.CASTable(score_astore).to_frame()
dm_scoreddf[dm_dec_target] = dm_scoreddf[dm_dec_target].astype(int)
trainData = dm_scoreddf[dm_scoreddf[dm_partitionvar]==dm_partition_train_val][[dm_dec_target, dm_predictionvar[1]]].rename(columns=lambda x:'0')
testData = dm_scoreddf[dm_scoreddf[dm_partitionvar]==dm_partition_test_val][[dm_dec_target, dm_predictionvar[1]]].rename(columns=lambda x:'0')
validData = dm_scoreddf[dm_scoreddf[dm_partitionvar]==dm_partition_validate_val][[dm_dec_target, dm_predictionvar[1]]].rename(columns=lambda x:'0')
trainData = pd.DataFrame(trainData)
testData = pd.DataFrame(testData)
validData = pd.DataFrame(validData)

#########################################
###  Register Model in Model Manager  ###
#########################################

import shutil
from sasctl import pzmm as pzmm
from sasctl import Session
from sasctl import register_model, publish_model
from sasctl._services.model_repository import ModelRepository as mr

### create session in cas
sess=Session(hostname, username=username, password=password, verify_ssl=False, protocol="http")

### create directories for metadata
output_path = Path(output_dir) / metadata_output_dir / model_name
if output_path.exists() and output_path.is_dir():
    shutil.rmtree(output_path)
os.makedirs(output_path)

### create metadata and import to model manager
pzmm.JSONFiles().calculateFitStat(trainData=trainData, testData=testData, validateData=validData, jPath=output_path)
pzmm.JSONFiles().generateROCLiftStat(dm_dec_target, int(dm_classtarget_level[1]), conn, trainData=trainData, testData=testData, validateData=validData, jPath=output_path)
file_list = os.listdir(output_path)
files = []
for i in file_list:
    new_dict = {'name':i, 'file':open(output_path / i)}
    files.append(new_dict)
with sess:
    reg_model = register_model(model_astore, model_name, project_name, files=files, force=True, version='latest')
#   pub_model = publish_model(model_name, 'maslocal')
#   score_example = pub_model.score(input1=1, input2=2, etc.)
