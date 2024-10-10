import math
import pickle
import pandas as pd
import numpy as np
from pathlib import Path

import settings

with open(Path(settings.pickle_path) / "[model_name].pickle", "rb") as pickle_model:
    model = pickle.load(pickle_model)

def score(input_vars_no_quotes_comma_separated):
    "Output: P_[target_var_name]0, P_[target_var_name]1, I_[target_var_name]"

    try:
        global model
    except NameError:
        with open(Path(settings.pickle_path) / "[model_name].pickle", "rb") as pickle_model:
            model = pickle.load(pickle_model)


    index=None
    if not isinstance(marital_status_single, pd.Series):
        index=[0]
    input_array = pd.DataFrame(
        {"input_var1": input_var1, "input_var2": input_var2, ...
        }, index=index
    )
    prediction = model.predict_proba(input_array).tolist()

    # Check for numpy values and convert to a CAS readable representation
    if isinstance(prediction, np.ndarray):
        prediction = prediction.tolist()

    if input_array.shape[0] == 1:
        if prediction[0][1] > 0.5:
            I_[target_var_name] = "1"
        else:
            I_[target_var_name] = "0"
        return prediction[0][0], prediction[0][1], I_[target_var_name]
    else:
        df = pd.DataFrame(prediction)
        proba_not = df[0]
        proba = df[1]
        classifications = np.where(df[1] > 0.5, '1', '0')
        return pd.DataFrame({'P_[target_var_name]0': proba_not, 'P_[target_var_name]1': proba, 'I_[target_var_name]': classifications})