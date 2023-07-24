Training script is named classification_model_xgboost.py.

Once published, scoring by api call is contained in the script classification_model_xgboost_score_by_api_call.py (adjustments are necessary for deployment environements).

Score code for 'core' XGBoost should include two extra lines to accommodate inputArray as a DMatrix.

The script xgboost_pythonScore.py includes these 2 lines of code.

The script xgboost_python_origScore.py is the score code that is generated automatically with sasctl/pzmm and does not include these 2 ilnes of code.
