/*################
  ### SAS CODE ###
  ################*/

cas casauto sessopts=(caslib=public, metrics=true, timeout=900);
libname cp cas sessref=casauto;

/*###########################
  ### Set Macro Variables ###
  ###########################*/

%let in_mem_tbl = 'aml_bank_prep';
%let caslib_ref = 'public';
%let libname_in_mem_tbl = public.aml_bank_prep;
%let target = 'ml_indicator';
%let predictedtarget = 'P_ml_indicator1';
%let inputs = {
			"checking_only_indicator", "prior_ctr_indicator", "address_change_2x_indicator",
			"cross_border_trx_indicator", "in_person_contact_indicator",
			"linkedin_indicator", "atm_deposit_indicator", "trx_10ksum_indicator",
			"common_merchant_indicator", "direct_deposit_indicator", "marital_status",
			"primary_transfer_cat", "citizenship_country_risk", "occupation_risk", 
			"credit_score", "distance_to_bank", "distance_to_employer", 
			"income", "num_acctbal_chgs_gt2000", "num_transactions"};
%let bias_var = "cross_border_trx_indicator";

/*#############################
  ### Identify Table in CAS ###
  #############################*/

proc cas;
		table.tableExists result=code /
			caslib=&caslib_ref, name=&in_mem_tbl;
			if code['exists'] <= 0 then do;
				table.loadTable /
					caslib=&caslib_ref,
					path=cats(&in_mem_tbl,'.sashdat'),
					casout={caslib=&caslib_ref,
						name=&in_mem_tbl,
						promote=TRUE};
			end;
		table.columnInfo result=col_list /
			table={caslib=&caslib_ref, name=&in_mem_tbl};
			describe col_list;
			print col_list.ColumnInfo[,'Column'];
run;

/*########################
  ### Model Parameters ###
  ########################*/

proc cas;
	decisionTree.gbtreeTrain /
		table={caslib=&caslib_ref, name=&in_mem_tbl}
		target=&target
		inputs=&inputs
		encodeName=TRUE
		nominals={&target}
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
		quantileBin=TRUE
		earlyStop={metric="MCR", stagnation=5, tolerance=0, minimum=FALSE,
					threshold=0, thresholdIter=0}
		casOut={caslib=&caslib_ref, name="casgradboost_model", replace=True}
		saveState={caslib=&caslib_ref, name="casgradboost_astore", replace=True}
    	;
run;

/*########################
  ### Write Score Code ###
  ########################*/

proc cas;

	decisionTree.gbtreeCode /
		modelTable={name="casgradboost_model"}
		code={casOut={caslib=&caslib_ref, name='casgradboost_scorecode', 
						replace=True, promote=False}}
	;
run;

/*########################
  ### Score New Table  ###
  ########################*/

proc cas;

	decisionTree.gbtreeScore /
		modelTable={name="casgradboost_model"}
		table={caslib=&caslib_ref, name=&in_mem_tbl}
		casOut={caslib=&caslib_ref, name='casgradboost_scored', replace=True}
		copyVars={&target}
		encodeName=TRUE
		assessOneRow=TRUE
	;
run;

/*########################
  ###   Assess Model   ###
  ########################*/

proc cas;
	percentile.assess /
		table={caslib=&caslib_ref, name="casgradboost_scored"}
		event="1"
		response=&target
		inputs=&predictedtarget
		cutStep=0.001
		casOut={caslib=&caslib_ref, name='casgradboost_assess', replace=True}
	;
run;

proc cas;
	dataStep.runCode /
		code = "data test;
				set public.casgradboost_assess_roc;
				if _KS_ = 1;
				run;";
run;

/* _KS2_ is the highest KS given cutSteps */
proc print data=cp.test;
run;

proc cas;
		table.tableExists result=code /
			caslib=&caslib_ref, name='casgradboost_assess_roc';
			if code['exists'] = 0 then do;
			print "The CAS table does not exist";
			end;

			if code['exists'] = 1 then do;
			print "The CAS table has a session scope";
			end;

			if code['exists'] = 2 then do;
			print "The CAS table has a global scope";
			end;
quit;


/*###################
  ### PYTHON CODE ###
  ###################*/

proc python;
submit;

###################
### Environment ###
###################

import swat
import pandas as pd

#############################
### Identify Table in CAS ###
#############################

### caslib and table to use in modeling
caslib = 'Public'
in_mem_tbl = 'FINANCIAL_SERVICES_PREP'
dm_inputdf = SAS.sd2df(str(caslib)+str('.')+str(in_mem_tbl))

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
