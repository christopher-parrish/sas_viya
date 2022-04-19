
import numpy as np
from scipy.stats import norm

def option_value(notional, vol, strike_price, spot_price, time_to_mat, risk_free_rate):
    "Output: option_val"
    d1 = (np.log(spot_price/strike_price) + (risk_free_rate+vol**2/2)*time_to_mat)/(vol*np.sqrt(time_to_mat))
    d2 = d1 - vol*np.sqrt(time_to_mat)
    option_price = norm.cdf(d1)*spot_price-norm.cdf(d2)*strike_price*np.exp(-risk_free_rate*time_to_mat)
    option_val = notional * option_price
    return float(option_val)
