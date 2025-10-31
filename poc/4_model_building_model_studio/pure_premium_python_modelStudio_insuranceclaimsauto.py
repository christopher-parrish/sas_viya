
########################
### Model Parameters ###
########################

### import python libraries
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, KBinsDiscretizer, FunctionTransformer, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.linear_model import TweedieRegressor, GammaRegressor, LinearRegression
from sklearn.feature_selection import RFE, RFECV
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_tweedie_deviance, d2_absolute_error_score, mean_absolute_error, mean_squared_error
from sklearn.utils import shuffle
from scipy.sparse import csr_matrix

### power param
# 0: Normal
# 1: Poisson
# (1,2): Compound Poisson Gamma
# 2: Gamma
# 3: Inverse Guassian

tweedie_params = {
             'power': 1.8, 
             'alpha': 0.1,
             'fit_intercept': True,
             'link': 'auto',
             'tol': 0.0001,
             'max_iter': 10000,
             'warm_start': False
             } 
print(tweedie_params)

gamma_params = {
             'alpha': 1,
             'fit_intercept': True,
             'tol': 0.0001,
             'max_iter': 100,
             'warm_start': False
             } 
print(gamma_params)

linear_params = {
             'fit_intercept': True,
             'copy_X': True,
             'n_jobs': None,
             'positive': False
             } 
print(linear_params)

### model manager information
model_name = 'tweedie_python'
project_name = 'Pure Premium'
description = 'Tweedie GLM'
model_type = 'GLM'
model_function = 'Prediction'
predict_syntax = 'predict'

### define macro variables for model
dm_dec_target = 'PurePremium'
dm_partitionvar = '_PartInd_'
create_new_partition = 'no' # 'yes', 'no'
dm_key = 'uniqueRecordID' 
dm_partition_validate_val, dm_partition_train_val, dm_partition_test_val = [0, 1, 2]
dm_partition_validate_perc, dm_partition_train_perc, dm_partition_test_perc = [0.3, 0.6, 0.1]
dm_predictionvar = [str('P_') + dm_dec_target]

### mlflow
use_mlflow = 'no' # 'yes', 'no'
mlflow_run_to_use = 0
mlflow_class_labels =['TENSOR']
mlflow_predict_syntax = 'predict'

### var to consider in bias assessment
bias_var = 'Gender'

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

####################
### Plot Columns ###
####################

from matplotlib import pyplot as plt

plt.hist(dm_inputdf[dm_dec_target])
plt.hist(dm_inputdf['Income'])

dm_inputdf.hist(figsize=(15,75), layout=(28,5))

##############################
### Final Modeling Columns ###
##############################

### transformations

dm_inputdf_raw = dm_inputdf

poly_cols_1 = [] # 'Age', 'Income'
poly_cols_1_out = [] # 'Age', 'Income', 'AgeSq', 'AgeIncome', 'IncomeSq'; 'bias_col' would be first if set to True
poly_1 = ('poly', PolynomialFeatures(degree=2, include_bias=False, interaction_only=False), poly_cols_1)

# poly_scale_cols_1 = []
# poly_scale_cols_1_out = []
# for i in poly_scale_cols_1:
#     poly_scale_cols_1_out.append(i+'Poly_Scale')
# poly_scale_1 = ('poly_scale', make_pipeline(poly_1, StandardScaler()), poly_scale_cols_1)

impute_cols_1 = []
impute_cols_1_out = []
for i in impute_cols_1:
    impute_cols_1_out.append(i+'Impute')
impute_1 = ('impute', SimpleImputer(strategy='most_frequent'), impute_cols_1) # 'median', 'mean'

encode_cols_1 = ['Rating_Category', 'Occupation', 'Marital_Status', 'Education', 'Gender', 'Car_Type', 'CarUse']
encode_cols_1_out = [] # transform output is in sorted order, so extra step is needed to align out column names 
for i in encode_cols_1:
    temp_list = sorted(dm_inputdf_raw[i].unique())
    for j in temp_list:
        encode_cols_1_out.append(i+j)
ohe_1 = ('encode', OneHotEncoder(sparse=False, handle_unknown='ignore'), encode_cols_1)

binned_cols_1 = ['Age', 'Car_Age', 'MotorVehicleRecordPoint', 'Travel_Time']
binned_cols_1_out = []
for i in binned_cols_1:
    binned_cols_1_out.append(i+'Bin')
bin_1 = ('bins', KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile',), binned_cols_1)

scaled_cols_1 = []
scaled_cols_1_out = []
for i in scaled_cols_1:
    scaled_cols_1_out.append(i+'Scale')
scale_1 = ('scales', StandardScaler(), scaled_cols_1)

log_cols_1 = []
log_cols_1_out = []
for i in log_cols_1:
    log_cols_1_out.append(i+'Log')
log_1 = ('log', FunctionTransformer(func=np.log), log_cols_1)

log_scale_cols_1 = ['Bluebook', 'Income']
log_scale_cols_1_out = []
for i in log_scale_cols_1:
    log_scale_cols_1_out.append(i+'Log_Scale')
log_scale_1 = ('log_scale', make_pipeline(FunctionTransformer(func=np.log), StandardScaler()), log_scale_cols_1)
### this performs multiple transforms on the same columns

keep_cols_1 = ['Exposure', 'DrivingUnderInfluence', 'Revoked', dm_key, dm_dec_target, dm_partitionvar]
keep_1 = ('keep', 'passthrough', keep_cols_1)

drop_cols_1 = ['Origination_Source']
drop_1 = ('drop', 'drop', drop_cols_1)

transforms = ColumnTransformer(transformers=[ohe_1, bin_1, log_scale_1, keep_1, drop_1], 
                                   remainder='drop', 
                                   verbose_feature_names_out=True)
                                    # remainder='passthrough'

df_temp = transforms.fit_transform(dm_inputdf_raw)
#transforms.get_feature_names_out() this does not work with some of the transformation - why?? no idea
### work around for column names (this needs to be done in the order of the transforms)
transform_cols = poly_cols_1_out + impute_cols_1_out + encode_cols_1_out + binned_cols_1_out + scaled_cols_1_out + log_cols_1_out + log_scale_cols_1_out + keep_cols_1
dm_inputdf = pd.DataFrame(data=df_temp, columns=transform_cols)

### create list of rejected predictor columns
dm_input = list(dm_inputdf.columns.values)
#dm_input = [x.replace(' ', '') for x in dm_input]
#dm_input = [x.replace('(', '_') for x in dm_input]
#dm_input = [x.replace(')', '_') for x in dm_input]
print(dm_input)
macro_vars = (dm_dec_target + ' ' + dm_partitionvar + ' ' + dm_key).split()
#string_cols = list(dm_inputdf.select_dtypes('object'))
#keep_predictors = [i for i in dm_input if i not in macro_vars]
#keep_predictors = [string_cols]
#rejected_predictors = [i for i in dm_input if i not in keep_predictors]
rejected_predictors = ['Rating_CategoryA', 'Occupation(missing)', 'Marital_StatusM',
                       'EducationBachelors', 'GenderF', 'Car_TypeHatchback', 'CarUseC', 'Exposure']
                        # 'Income', 'IncomeSq', 
rejected_vars = rejected_predictors + macro_vars
for i in rejected_vars:
    dm_input.remove(i)
print(dm_input)

##################
### Data Split ###
##################

### create train, test, validate datasets using existing partition column
dm_traindf = dm_inputdf[dm_inputdf[dm_partitionvar] == dm_partition_train_val]
dm_testdf = dm_inputdf.loc[(dm_inputdf[dm_partitionvar] == dm_partition_test_val)]
dm_validdf = dm_inputdf.loc[(dm_inputdf[dm_partitionvar] == dm_partition_validate_val)]
y_train = dm_traindf[dm_dec_target]
y_test = dm_testdf[dm_dec_target]
y_valid = dm_validdf[dm_dec_target]
fullX = dm_inputdf.loc[:, dm_input]
fully = dm_inputdf[dm_dec_target]

##########################
### Variable Selection ###
##########################

### Recursive Feature Elimination (RFE) with Crossvalidation (auto-select number of variables)
models_for_rfe = [DecisionTreeRegressor(), GradientBoostingRegressor()] #RandomForestRegressor() 
rfe_cols_cv = []
for i in models_for_rfe:
    rfe_cv = RFECV(estimator=i, step=1, cv=10, min_features_to_select=1)
    rfe_cv.fit(fullX,fully)
    rfe_cols_cv.append(list(rfe_cv.get_feature_names_out()))
    
#####################
### Training Code ###
#####################

models_for_training_list = [TweedieRegressor(**tweedie_params)]
model_results_list = []
model_list = []

for i in models_for_training_list:
    for j in range(0, len(rfe_cols_cv)):
        X_train = dm_traindf.loc[:, rfe_cols_cv[j]]
        X_test = dm_testdf.loc[:, rfe_cols_cv[j]]
        X_valid = dm_validdf.loc[:, rfe_cols_cv[j]]
        dm_model = i
        dm_model.fit(X_train, y_train, sample_weight=dm_traindf['Exposure'])
        #cross_val_score(dm_model, X_train, y_train, cv=10, n_jobs=1)
        score = dm_model.score(X_valid, y_valid)
        model_results_list.append(score)
        name = [str(i)[0:10]+str('_varlist')+str(j)]
        model_list.append(name)
        print('%s %.4f' % (name, score))

# models = dict('LinReg',LinearRegression(**linear_params), 'GammReg', GammaRegressor(**gamma_params), 'TweedieReg', TweedieRegressor(**tweedie_params))        
# sparse_matrix = csr_matrix(dm_traindf.loc[:, rfe_cols_cv[j]])

###################################
###  Score Data & Assess Model  ###
###################################

def score_func(partition_df, partition_X, partition_y, partition):
    dfProba = pd.DataFrame(pd.concat([partition_X.reset_index(drop=True), 
                            partition_y.reset_index(drop=True), 
                            partition_df['Exposure'].reset_index(drop=True), 
                            pd.Series(data=dm_model.predict(partition_X), name='Prediction')],
                            axis=1)
                           )
    dfProba['Predicted_Claims'] = dfProba['Exposure']*dfProba['Prediction']
    observed_claims = np.sum(dfProba['Exposure']*dfProba['PurePremium'])
    predicted_claims = np.sum(dfProba['Predicted_Claims'])
    diff_predicted_minus_observed = predicted_claims-observed_claims
    perc_diff = diff_predicted_minus_observed/observed_claims
    print('**********')
    print(partition)
    print('**********')
    print('observed_claims:', "${:0,.2f}".format(observed_claims))
    print('predicted_claims', "${:0,.2f}".format(predicted_claims))
    print('diff_observed_minus_predicted:', "${:0,.2f}".format(diff_predicted_minus_observed))
    print('%_diff_of_observed:', "{0:.0%}".format(perc_diff))
    print('% variance explained:', "{0:.0%}".format(dm_model.score(partition_X, partition_y)))
    print('mean observed:', "${:0,.2f}".format(np.mean(partition_y)))
    print('mean predicted:', "${:0,.2f}".format(np.mean(dfProba['Prediction'])))
    print('mean tweedie deviance:', "${:0,.2f}".format(mean_tweedie_deviance(partition_y, dfProba['Prediction'], power=tweedie_params['power'])))
    print('d2_absolute error:', "${:0,.2f}".format(d2_absolute_error_score(partition_y, dfProba['Prediction'])))
    #print('mean absolute error:', "${:0,.2f}".format(mean_absolute_error(partition_y, dfProba['Prediction'])))
    #print('mean squared error:', "${:0,.2f}".format(mean_squared_error(partition_y, dfProba['Prediction'])))
    #print('root mean squared error:', "${:0,.2f}".format(np.sqrt(mean_squared_error(partition_y, dfProba['Prediction']))))
    global df
    df = pd.DataFrame(dfProba)

dm_scoreddf = df
score_func(dm_traindf, X_train, y_train, 'train')
trainProba = df
trainData = trainProba[[dm_dec_target, 'Prediction']]
# score_func(dm_testdf, X_test, y_test, 'test')
# testProba = df
# testData = testProba[[dm_dec_target, 'Prediction']]
score_func(dm_validdf, X_valid, y_valid, 'validate')
validProba = df
validData = validProba[[dm_dec_target, 'Prediction']]
