
#####################################################
###       Train & Publish SAS gbTree Model        ###
#####################################################

###################
### Credentials ###
###################

import keyring
import runpy
import os

### run script that contains username, password, hostname, working directory, and output directory
    ### ...OR define directly in this script
from password import hostname, port, wd, output_dir
runpy.run_path(path_name='password.py')
username = keyring.get_password('cas', 'username')
password = keyring.get_password('cas', username)
metadata_output_dir = 'outputs'

###################
### Environment ###
###################

import swat

conn =  swat.CAS(hostname, port, username=username, password=password, protocol='cas')
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
m=20
seed=12345
nTree=100
learningRate=0.1
subSampleRate=0.5
lasso=0
ridge=1
distribution="binary"
maxBranch=2
maxLevel=5
leafSize=5
missing="useinsearch"
minUseInSearch=1
nBins=50
quantileBin=True

early_stop_params = dict(
    metric="MCR",
    stagnation=5,
    tolerance=0,
    minimum=False,
    threshold=0,
    thresholdIter=0
    )

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
dm_model = conn.decisionTree.gbtreeTrain(
    earlyStop=early_stop_params,
    table=dict(caslib=caslib, name=in_mem_tbl, where=train_part),
    target=dm_dec_target,
    nominal=dm_dec_target,
    inputs=dm_input,
    encodeName=True,
    casOut=dict(caslib=caslib, name=cas_out_tbl, replace=True),
    saveState=dict(caslib=caslib, name=astore_tbl, replace=True),
    m=m, seed=seed, nTree=nTree, learningRate=learningRate, subSampleRate=subSampleRate, 
    lasso=lasso, ridge=ridge, distribution=distribution, maxBranch=maxBranch, 
    maxLevel=maxLevel, leafSize=leafSize, missing=missing, minUseInSearch=minUseInSearch, 
    nBins=nBins, quantileBin=quantileBin 
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
  inputs=dm_predictionvar[2],
  cutStep=0.0001,
  casOut=dict(caslib=caslib, name='gbtree_python_assess', replace=True)
  )

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

#################################
### Publish Model in SAS Viya ###
#################################

conn.modelPublishing.publishModel(
  modelName=model_name,
  storeTables=dict(caslib=caslib, name=astore_tbl),
  modelType="DS2"
  )