########################
### Model Parameters ###
########################

### Notes ###
# keep the score code as simple as possible to avoid errors
# "Output:..." statement vars need to match score outputs in node results
# columns can be printed at the end of the training script or in a separate node
    # insert_columns_commas_no_quote = print((', '.join(dm_input)))
    # insert_columns_commas = print(dm_input)
    # "Output:..." = string_line = str("Output: ") + str(dm_predictionvar[0]) + str(", ") + str(dm_predictionvar[1]) + str(", ") + str(dm_classtarget_intovar)

# import python libraries
import pickle
import numpy as np
import pandas as pd

##################
### Score Code ###
##################

with open(settings.pickle_path + dm_pklname, 'rb') as f:
	model = pickle.load(f)

def score_method(net_worth, credit_score, num_dependents, at_current_job_1_year, credit_history_mos, job_in_education, num_transactions, debt_to_income, amount, gender, age, job_in_hospitality):
	"Output: P_event_indicator0, P_event_indicator1, I_event_indicator"
	df = pd.DataFrame([[net_worth, credit_score, num_dependents, at_current_job_1_year, credit_history_mos, job_in_education, num_transactions, debt_to_income, amount, gender, age, job_in_hospitality]], 
	columns=['net_worth',
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
    'job_in_hospitality'])
	df_pred_prob = model.predict_proba(df)
	df_pred = model.predict(df)
	return float(df_pred_prob[0][0]), float(df_pred_prob[0][1]), float(df_pred[0])
