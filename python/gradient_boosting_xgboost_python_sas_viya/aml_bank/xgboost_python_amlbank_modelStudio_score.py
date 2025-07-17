########################
### Model Parameters ###
########################

### Notes ###
# the ORDER of the columns in the function need to match those exactly as shown in the output pane in Model Studio (very important)
# may receive an error in Model Manager scoring / performance 'DS2 "pymas" package encountered a failure in the 'execute' method' if order does not match pickle
# can check pickle feature order by using model.feature_names_in_ (may be different based on modeling object)
# the columns can be printed at the end of the training script or in a separate node
    # insert_columns_commas_no_quote = print((', '.join(dm_input)))
    # insert_columns_commas = print(dm_input)
    # "Output:..." = string_line = str("Output: ") + str(dm_predictionvar[0]) + str(", ") + str(dm_predictionvar[1]) + str(", ") + str(dm_classtarget_intovar)
# "Output:..." statement vars need to match score outputs in node results

# import python libraries
import pickle
import numpy as np
import pandas as pd

##################
### Score Code ###
##################

with open(settings.pickle_path + dm_pklname, 'rb') as f:
	model = pickle.load(f)

### download pickle file and check to see if code works, if necessary ###	
# with open('C:/Users/chparr/Downloads/_9P3G8SB585B2PYEIGNTB7XS95_PKL.pickle', 'rb') as f:
# 	model = pickle.load(f)

def score_method(num_transactions, credit_score, marital_status_single, checking_only_indicator, prior_ctr_indicator, address_change_2x_indicator, cross_border_trx_indicator, in_person_contact_indicator, linkedin_indicator, trx_10ksum_indicator, common_merchant_indicator, direct_deposit_indicator, primary_transfer_cash):
	"Output: P_ml_indicator0, P_ml_indicator1, I_ml_indicator"
	df = pd.DataFrame([[num_transactions, credit_score, marital_status_single, checking_only_indicator, prior_ctr_indicator, address_change_2x_indicator, cross_border_trx_indicator, in_person_contact_indicator, linkedin_indicator, trx_10ksum_indicator, common_merchant_indicator, direct_deposit_indicator, primary_transfer_cash]], 
    columns=['num_transactions',
			 'credit_score',
			 'marital_status_single', 
			 'checking_only_indicator', 
             'prior_ctr_indicator', 
             'address_change_2x_indicator', 
             'cross_border_trx_indicator', 
             'in_person_contact_indicator', 
             'linkedin_indicator', 
             'trx_10ksum_indicator', 
             'common_merchant_indicator', 
             'direct_deposit_indicator',             
             'primary_transfer_cash'              
             ])
	df_pred_prob = model.predict_proba(df)
	df_pred = model.predict(df)
	return float(df_pred_prob[0][0]), float(df_pred_prob[0][1]), float(df_pred[0])

### score 1 row ###
# score_method(-0.596972665,0.3220850945,0,0,1,1,1,0,0,0,1,1,0)