
#################################################################
###    Train XGBoost Model in SAS Viya's Build Models App     ###
#################################################################

########################
### Model Parameters ###
########################

# import python libraries
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import shuffle
import pickle

### model arugments
xgb_params = dict(
    base_score=0.5,
    booster='gbtree',
    eval_metric='auc',
    colsample_bytree=1,
    colsample_bylevel=1,
    colsample_bynode=1,
    gamma=0,
    grow_policy='depthwise',
    learning_rate=0.1,
    max_bin=256,
    max_delta_step=0,
    max_depth=3,
    max_leaves=0,
    min_child_weight=1,
    nthread=None,
    num_parallel_tree=1,
    objective='binary:logistic',
    predictor='auto',
    process_type='default',
    refresh_leaf=1,
    reg_alpha=0,
    reg_lambda=1,
    sampling_method='uniform',
    scale_pos_weight=1,
    seed=None,
    seed_per_iteration=False,
    sketch_eps=0.03,
    subsample=1,
    tree_method='auto'
    )
print(xgb_params)

### XGBOOST CORE
xgb_params_train = {'num_boost_round': 100,
                    'early_stopping_rounds': 50}
cutoff = 0.05

### model manager information
model_name = 'xgboost_python_modelStudio'
project_name = 'AML Risk Score'
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
dm_model = xgb.DMatrix(X_train, y_train)
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

####################
### Pickle Model ###
####################

with open(dm_pklpath, 'wb') as f:
	pickle.dump(dm_model, f)
