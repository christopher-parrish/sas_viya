/* Data prep for table to use in Python model development */

/* create work table from CAS table */
data financial_services;
    set public.financial_services;
run;

/* calculate correlations */
proc corr data=financial_services;
    var event_indicator gender ever_missed_obligation at_current_job_1_year smoker uses_direct_deposit business_owner homeowner marital_status num_dependents job_industry citizenship region years_at_residence age debt_to_income num_transactions credit_history_mos credit_score amount net_worth;
run;

/* standardize continuous variables */
proc stdize data=financial_services method=std nomiss out=financial_services_out;
	var age amount credit_history_mos credit_score debt_to_income net_worth 
		num_transactions years_at_residence;
run;

/* data engineering / bucketing */
data financial_services_out;
	set financial_services_out;
job_in_education = 0;
job_in_energy = 0;
job_in_media = 0;
job_in_transport = 0;
job_in_hospitality = 0;
job_in_healthcare = 0;
job_in_manufacturing = 0;
job_in_financial = 0;

if job_industry = 0
	then job_in_education = 1;
	else if job_industry = 1
		then job_in_energy = 1;
		else if job_industry = 2
			then job_in_media =1;
			else if job_industry = 3
				then job_in_transport = 1;
				else if job_industry = 4
					then job_in_hospitality = 1;
					else if job_industry = 5
						then job_in_healthcare = 1;
						else if job_industry = 6
							then job_in_manufacturing = 1;
							else if job_industry = 7
								then job_in_financial = 1;

citizenship_us = 0;
citizenship_latam = 0;
citizenship_eu = 0;
citizenship_middle_east = 0;
citizenship_africa = 0;
citizenship_india = 0;
citizenship_china = 0;
citizenship_other = 0;

if citizenship = 0
	then citizenship_us = 1;
	else if citizenship = 1
		then citizenship_latam = 1;
		else if citizenship = 2
			then citizenship_eu = 1;
			else if citizenship = 3
				then citizenship_middle_east = 1;
				else if citizenship = 4
					then citizenship_africa = 1;
					else if citizenship = 5
						then citizenship_india = 1;
						else if citizenship = 6
							then citizenship_china = 1;
							else if citizenship = 7
								then citizenship_other = 1;

region_ca = 0;
region_ny = 0;
region_fl = 0;
region_tx = 0;
region_ne = 0;
region_so = 0;
region_mw = 0;
region_we = 0;

if region = 0
	then region_ca = 1;
	else if region = 1
		then region_ny = 1;
		else if region = 2
			then region_fl = 1;
			else if region = 3
				then region_tx = 1;
				else if region = 4
					then region_ne = 1;
					else if region = 5
						then region_so = 1;
						else if region = 6
							then region_mw = 1;
							else if region = 7
								then region_we = 1;

marital_status_single = 0;
marital_status_married = 0;
marital_status_divorced = 0;

if marital_status = 0
	then marital_status_single = 1;
	else if marital_status = 1
		then marital_status_married = 1;
		else if marital_status = 2
			then marital_status_divorced = 1;
run;

/* create train, test, validate partitions and verify*/
proc partition data=financial_services_out samppct=30 samppct2=60 seed=10 partind nthreads=3;
   output out=financial_services_out(rename=(_partind_=analytic_partition));
run;

proc freq data=financial_services_out;
    tables analytic_partition;
run;

/* select columns to use in modeling */
proc sql;
    create table financial_services_prep as
   select
      account_id
         label = "unique identifier",
      event_indicator
         label = "indicator for financial services event",
      gender
         label = "indicator for gender (0) male, (1) female",
      ever_missed_obligation
         label = "indicator for financial services customer ever missed a required obligation over the past 10 years",
      at_current_job_1_year
         label = "indicator for financial services customer at their current job for at least 1 year",
      smoker
         label = "indicator for financial services customer is a smoker",
      uses_direct_deposit
         label = "indicator for financial services customer uses direct deposit",
      business_owner
         label = "indicator for financial services customer is a business owner",
      homeowner
         label = "indicator for financial services customer is a homeowner",
      num_dependents
         label = "number of children in financial services customer's household",
      years_at_residence
         label = "number of years financial services customer has lived in current residence",
      age
         label = "age of financial services customer",
      debt_to_income
         label = "debt to income ratio of financial services customer",
      num_transactions
         label = "number of transactions financial services customer has made over the past year in account",
      credit_history_mos
         label = "number of months of financial services customer's credit history",
      credit_score
         label = "credit score of financial services customer",
      amount
         label = "amount associated with financial services customer's account",
      net_worth
         label = "net worth of financial services customer",
      id_important_activity
         label = "indicator for a recent important activity involving financial services customer for decisioning rule",
      id_direct_contact
         label = "indicator for recent contact with financial services customer for decisioning rule",
      id_current_fs_relationship,
      job_in_education,
      job_in_energy,
      job_in_media,
      job_in_transport,
      job_in_hospitality,
      job_in_healthcare,
      job_in_manufacturing,
      job_in_financial,
      citizenship_us,
      citizenship_latam,
      citizenship_eu,
      citizenship_middle_east,
      citizenship_africa,
      citizenship_india,
      citizenship_china,
      citizenship_other,
      region_ca,
      region_ny,
      region_fl,
      region_tx,
      region_ne,
      region_so,
      region_mw,
      region_we,
      marital_status_single,
      marital_status_married,
      marital_status_divorced,
      analytic_partition
   from financial_services_out;
run;

/* using training table, convert to table generative above to Pandas dataframe and begin modeling in Python */

proc python;
submit;

import pandas as pd

### table to use in modeling
in_mem_tbl = 'FINANCIAL_SERVICES_PREP'
dm_inputdf = SAS.sd2df(in_mem_tbl)

### print columns for review of model parameters
print(dm_inputdf.dtypes)

########################
### Model Parameters ###
########################

# import python libraries
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import shuffle

### model arugments
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

### model manager information
model_name = 'logit_python_finsvcs_SASStudio'
project_name = 'Financial Services'
description = 'Logit Python'
model_type = 'logistic_regression'
predict_syntax = 'predict_proba'

### define macro variables for model
dm_dec_target = 'event_indicator'
dm_partitionvar = 'analytic_partition'
create_new_partition = 'no' # 'yes', 'no'
dm_key = 'account_id' 
dm_classtarget_level = ['0', '1']
dm_partition_validate_val, dm_partition_train_val, dm_partition_test_val = [0, 1, 2]
dm_partition_validate_perc, dm_partition_train_perc, dm_partition_test_perc = [0.3, 0.6, 0.1]

### create list of regressors
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
#rejected_predictors = []

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

### remove added index column in Model Studio
#dm_inputdf.drop('_dmIndex_', axis=1, inplace=True)

### create list of model variables
dm_input = list(dm_inputdf.columns.values)
macro_vars = (dm_dec_target + ' ' + dm_partitionvar + ' ' + dm_key).split()
rejected_predictors = [i for i in dm_input if i not in keep_predictors]
rejected_vars = rejected_predictors # + macro_vars (include macro_vars if rejected_predictors are explicitly listed - not contra keep_predictors)
for i in rejected_vars:
    dm_input.remove(i)
print(dm_input)

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
print('model_parameters')
print(dm_model)
print(' ')
print('model_performance')
print('score_test:', dm_model.score(X_test, y_test))
print('score_valid:', dm_model.score(X_valid, y_valid))
print('confusion_matrix:')
print('(tn, fp, fn, tp)')
print((tn, fp, fn, tp))
print('classification_report:')
print(classification_report(y_test, predictions))
if model_type == 'logistic_regression':
    orat = np.exp(dm_model.coef_, out=None)
    c1 = np.vstack([predictors,orat])
    c2 = np.transpose(c1)
    c3 = pd.DataFrame(c2, columns=['predictors', 'odds_ratio'])
    print('intercept:')
    print(dm_model.intercept_)
    print('odds_ratios:')
    print(c3)

endsubmit;
run;
quit;
