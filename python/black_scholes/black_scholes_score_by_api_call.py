
###################
### Credentials ###
###################

import keyring
import runpy
import os

### run script that contains username, password, and hostname_model
    ### ...OR define directly in this script
wd = 'C:/...'
os.chdir(wd)
from password import hostname_model
runpy.run_path(path_name='password.py')
username = keyring.get_password('cas', 'username')
password = keyring.get_password('cas', username)

#################
### Get Token ###
#################

from requests import request

url = hostname_model + '/SASLogon/oauth/token' 
r = request('POST', url,
            data='grant_type=password&username=%s&password=%s' %(username, password),
            headers={
                'Accept': 'application/json',
                'Content-Type': 'application/x-www-form-urlencoded'
            },
            auth=('sas.ec', ''),
            verify=False)
token = r.json()['access_token']

############################
### View Deployed Models ###
############################

headers = {'Authorization': 'Bearer ' + token}
url = hostname_model + '/microanalyticScore/modules/'
r = request('GET', url, params={}, headers=headers, verify=False)
r.json()

###########################
### Select Target Model ###
###########################

model_name = 'black_scholes_python' # case sensitive

headers = {'Authorization': 'Bearer ' + token}
url = hostname_model + '/microanalyticScore/modules/' + model_name + '/steps'
r = request('GET', url, params={}, headers=headers, verify=False)
r.json()

### note: the output contains the 'inputs' and 'outputs' metadata associated with the scoring

##########################
### Score Target Model ###
##########################

def score_model(notional, vol, strike_price, spot_price, time_to_mat, risk_free_rate):
    data = '{"inputs":[ {"name":"notional", "value":' + str(notional) + '}, \
                    {"name":"risk_free_rate", "value":' + str(risk_free_rate) + '}, \
                    {"name":"spot_price", "value":' + str(spot_price) + '}, \
                    {"name":"strike_price", "value":' + str(strike_price) + '}, \
                    {"name":"time_to_mat", "value":' + str(time_to_mat) + '}, \
                    {"name":"vol", "value":' + str(vol) + '} ]}'
    headers = {'Content-Type': 'application/vnd.sas.microanalytic.module.step.input+json', 
               'Authorization': 'Bearer ' + token}
    url = hostname_model + '/microanalyticScore/modules/' + model_name + '/steps/score'
    r = request('POST', url, data=data, headers=headers, verify=False)
    return_val = (r.json()['outputs'][1]['value'])
    return return_val

score_val = "${0:.2f}".format(score_model(200, .5, 15, 5, 1, 1))
print ('')
print('****************************************************************************')
print("This option position has a " + str(score_val) + " value")
print('****************************************************************************')