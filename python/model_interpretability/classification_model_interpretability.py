
##############################################
###       SAS Model Interpretability       ###
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
from password_poc import hostname, wd
runpy.run_path(path_name='password_poc.py')
username = keyring.get_password('cas', 'username')
password = keyring.get_password('cas', username)
metadata_output_dir = 'outputs'

###################
### Environment ###
###################

import swat

port = 443
os.environ['CAS_CLIENT_SSL_CA_LIST']=str(wd)+str('/ca_cert_poc.pem')
conn =  swat.CAS(hostname, port, username=username, password=password, protocol='http')
#print(conn)
#print(conn.serverstatus())

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

#dm_inputdf = pd.read_csv(str(data_dir)+str('/')+in_mem_tbl+str('.csv'))

### print columns for review of model parameters
print(dm_inputdf.dtypes)

########################
### Model Parameters ###
########################

### import python libraries
import numpy as np
from sklearn.utils import shuffle

### import actionsets
conn.loadactionset('decisionTree')
conn.loadactionset('astore')
conn.loadactionset('explainModel')
conn.loadactionset('transpose')

### var to consider in partial dependency
pd_var1 = 'cross_border_trx_indicator'
pd_var2 = 'trx_10ksum_indicator'

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
model_name = 'gradboost_cas'
project_name = 'Risk Score'
description = 'GradBoost CAS'
model_type = 'gradient_boost'

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

###########################################
### Upload Final Modeling Table to Viya ###
###########################################

#dm_inputdf_temp = conn.upload(dm_inputdf).casTable

##################
### Data Split ###
##################

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

sample_size = 500
rand_obs = np.array(dm_inputdf[dm_key].sample(n=sample_size, random_state=12345).as_matrix())

for i in range(sample_size):
    obs_num = rand_obs[i].astype(int).item()
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
