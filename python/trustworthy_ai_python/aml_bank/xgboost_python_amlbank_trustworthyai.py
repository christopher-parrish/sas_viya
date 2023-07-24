
##############################################
###           SAS Fair AI Tools            ###
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
    ### ...OR define directly in this script
from password import hostname, port, protocol, wd
runpy.run_path(os.path.join(wd, 'password_poc.py'))
username = keyring.get_password('cas', 'username')
password = keyring.get_password('cas', username)

###################
### Environment ###
###################

import swat
import pandas as pd

conn =  swat.CAS(hostname=hostname, port=port, username=username, password=password, protocol=protocol)
print(conn)

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
import xgboost as xgb
from sklearn.utils import shuffle

### import actionsets
conn.loadactionset('fairAITools')

### var to consider in bias assessment
bias_var = 'cross_border_trx_indicator'

### model arugments
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

#####################
### Training Code ###
#####################

### estimate & fit model
dm_model = xgb.XGBClassifier(**xgb_params)
dm_model.fit(X_train, y_train)

### score full data
fullX = dm_inputdf.loc[:, dm_input]
fully = dm_inputdf[dm_dec_target]
dm_scoreddf_prob = pd.DataFrame(dm_model.predict_proba(fullX), columns=dm_predictionvar)
dm_scoreddf_class = pd.DataFrame(dm_model.predict(fullX), columns=[dm_classtarget_intovar])
dm_scoreddf_bias = pd.DataFrame(dm_inputdf, columns=[dm_dec_target, bias_var])
dm_scoreddf = pd.concat([dm_scoreddf_prob, dm_scoreddf_class], axis=1)
scored = pd.concat([dm_scoreddf, dm_scoreddf_bias], axis=1)
scored_data = conn.upload(scored).casTable

##########################
###   Fair AI Tools    ###
### Assess Bias Action ###
##########################

conn.fairAITools.assessBias(
		table = scored_data,
		modelTableType = 'NONE',
		response = dm_dec_target,
		predictedVariables = dm_predictionvar,
		responseLevels = dm_classtarget_level,
		sensitiveVariable = bias_var
        )
