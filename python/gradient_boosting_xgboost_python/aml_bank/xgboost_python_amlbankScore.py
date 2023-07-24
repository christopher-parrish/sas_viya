

import math
import pickle
import pandas as pd
import numpy as np
import xgboost as xgb

import settings

with open(settings.pickle_path + 'xgboost_python.pickle', 'rb') as _pFile:
    _thisModelFit = pickle.load(_pFile)

def scorexgboost_python(checking_only_indicator, prior_ctr_indicator, address_change_2x_indicator, cross_border_trx_indicator, in_person_contact_indicator, linkedin_indicator, trx_10ksum_indicator, common_merchant_indicator, direct_deposit_indicator, marital_status, primary_transfer_cat, credit_score, num_transactions):
    "Output: EM_EVENTPROBABILITY, EM_CLASSIFICATION"

    try:
        global _thisModelFit
    except NameError:

        with open(settings.pickle_path + 'xgboost_python.pickle', 'rb') as _pFile:
            _thisModelFit = pickle.load(_pFile)

    try:
        inputArray = pd.DataFrame([[checking_only_indicator, prior_ctr_indicator, address_change_2x_indicator, cross_border_trx_indicator, in_person_contact_indicator, linkedin_indicator, trx_10ksum_indicator, common_merchant_indicator, direct_deposit_indicator, marital_status, primary_transfer_cat, credit_score, num_transactions]],
                                  columns=['checking_only_indicator', 'prior_ctr_indicator', 'address_change_2x_indicator', 'cross_border_trx_indicator', 'in_person_contact_indicator', 'linkedin_indicator', 'trx_10ksum_indicator', 'common_merchant_indicator', 'direct_deposit_indicator', 'marital_status', 'primary_transfer_cat', 'credit_score', 'num_transactions'],
                                  dtype=float)
        inputArray = xgb.DMatrix(inputArray)
        prediction = _thisModelFit.predict(inputArray)
    except ValueError:
    # For models requiring or including an intercept value, a 'const' column is required
    # For example, many statsmodels models include an intercept value that must be included for the model prediction
        inputArray = pd.DataFrame([[1.0, checking_only_indicator, prior_ctr_indicator, address_change_2x_indicator, cross_border_trx_indicator, in_person_contact_indicator, linkedin_indicator, trx_10ksum_indicator, common_merchant_indicator, direct_deposit_indicator, marital_status, primary_transfer_cat, credit_score, num_transactions]],
                                columns=['const', 'checking_only_indicator', 'prior_ctr_indicator', 'address_change_2x_indicator', 'cross_border_trx_indicator', 'in_person_contact_indicator', 'linkedin_indicator', 'trx_10ksum_indicator', 'common_merchant_indicator', 'direct_deposit_indicator', 'marital_status', 'primary_transfer_cat', 'credit_score', 'num_transactions'],
                                dtype=float)
        inputArray = xgb.DMatrix(inputArray)
        prediction = _thisModelFit.predict(inputArray)

    try:
        EM_EVENTPROBABILITY = float(prediction)
    except TypeError:
    # If the model expects non-binary responses, a TypeError will be raised.
    # The except block shifts the prediction to accept a non-binary response.
        EM_EVENTPROBABILITY = float(prediction[:,1])

    if (EM_EVENTPROBABILITY >= 0.04836266169444121):
        EM_CLASSIFICATION = '1'
    else:
        EM_CLASSIFICATION = '0' 

    return(EM_EVENTPROBABILITY, EM_CLASSIFICATION)
