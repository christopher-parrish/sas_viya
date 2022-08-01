
#####################################################
###     Train & Register CAS logistic Model       ###
#####################################################

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

# import python libraries
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.utils import shuffle
### import actionsets
conn.loadactionset('regression')
conn.loadactionset('astore')

### model arugments
event = 'LAST'
selection_method = 'STEPWISE'

### model manager information
model_name = 'logit_cas'
project_name = 'Risk Score'
description = 'Logit CAS'
model_type = 'Logistic Regression'
metadata_output_dir = 'outputs'

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

#####################
### Training Code ###
#####################

### estimate & fit model
dm_model = conn.regression.logistic(
    model={"depVars":[{"name":dm_dec_target, "options":{"event":event}}], "effects":[{"vars":dm_input}],"informative":True },
    partByVar={"name":dm_partitionvar, "train":str(dm_partition_train_val), "valid":str(dm_partition_validate_val),"test":str(dm_partition_test_val)},
    output={"casOut": {"name":cas_out_tbl, "caslib":caslib, "replace":True}, "copyVars":{dm_dec_target}, "into":dm_classtarget_intovar},
    selection={"method":selection_method},
    table={"name":in_mem_tbl, "caslib":caslib},
    store={"name":astore_tbl, "caslib":caslib, "replace":True})

### score full data
conn.astore.score(
    table={"name":in_mem_tbl, "caslib":caslib}, 
    copyvars=[dm_dec_target, dm_partitionvar],
    casout={"name":cas_score_tbl, "replace":True},
    rstore={"name":astore_tbl, "caslib":caslib})
score_astore = conn.CASTable(cas_score_tbl)

### create tables with predicted values
dm_scoreddf = conn.CASTable(score_astore).to_frame()
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
print(conn.astore.describe(rstore={"name":astore_tbl, "caslib":caslib}, epcode=True).Description)
print(conn.astore.describe(rstore={"name":astore_tbl, "caslib":caslib}, epcode=True).InputVariables)
print(conn.astore.describe(rstore={"name":astore_tbl, "caslib":caslib}, epcode=True).OutputVariables)
print(conn.astore.describe(rstore={"name":astore_tbl, "caslib":caslib}, epcode=True).epcode)
model_astore = conn.CASTable(astore_tbl, caslib=caslib)

#######################################
### Register Model in Model Manager ###
## Ensure Model Does Not Exist in MM ##
##### Using PZMM Zips Up Metadata #####
#######################################

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