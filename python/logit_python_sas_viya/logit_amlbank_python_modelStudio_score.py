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

def score_method(checking_only_indicator, prior_ctr_indicator, address_change_2x_indicator, cross_border_trx_indicator, in_person_contact_indicator, linkedin_indicator, trx_10ksum_indicator, common_merchant_indicator, direct_deposit_indicator, marital_status, primary_transfer_cat, credit_score, num_transactions):
	"Output: P_ml_indicator0, P_ml_indicator1, I_ml_indicator"
	df = pd.DataFrame([[checking_only_indicator, prior_ctr_indicator, address_change_2x_indicator, cross_border_trx_indicator, in_person_contact_indicator, linkedin_indicator, trx_10ksum_indicator, common_merchant_indicator, direct_deposit_indicator, marital_status, primary_transfer_cat, credit_score, num_transactions]], columns=['checking_only_indicator', 'prior_ctr_indicator', 'address_change_2x_indicator', 'cross_border_trx_indicator', 'in_person_contact_indicator', 'linkedin_indicator', 'trx_10ksum_indicator', 'common_merchant_indicator', 'direct_deposit_indicator', 'marital_status', 'primary_transfer_cat', 'credit_score', 'num_transactions'])
	df_pred_prob = model.predict_proba(df)
	df_pred = model.predict(df)
	return float(df_pred_prob[0][0]), float(df_pred_prob[0][1]), float(df_pred[0])