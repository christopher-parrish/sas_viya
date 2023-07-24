
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

model_name = 'xgboost_python' # case sensitive

headers = {'Authorization': 'Bearer ' + token}
url = hostname_model + '/microanalyticScore/modules/' + model_name + '/steps'
r = request('GET', url, params={}, headers=headers, verify=False)
r.json()

### note: the output contains the 'inputs' and 'outputs' metadata associated with the scoring

##########################
### Score Target Model ###
##########################

def score_model(address_change_2x_indicator, checking_only_indicator, common_merchant_indicator, credit_score, cross_border_trx_indicator, direct_deposit_indicator, in_person_contact_indicator, linkedin_indicator, marital_status, num_transactions, primary_transfer_cat, prior_ctr_indicator, trx_10ksum_indicator):
    data = '{"inputs":[ {"name":"address_change_2x_indicator", "value":' + str(address_change_2x_indicator) + '}, \
                    {"name":"checking_only_indicator", "value":' + str(checking_only_indicator) + '}, \
                    {"name":"common_merchant_indicator", "value":' + str(common_merchant_indicator) + '}, \
                    {"name":"credit_score", "value":' + str(credit_score) + '}, \
                    {"name":"cross_border_trx_indicator", "value":' + str(cross_border_trx_indicator) + '}, \
                    {"name":"direct_deposit_indicator", "value":' + str(direct_deposit_indicator) + '}, \
                    {"name":"in_person_contact_indicator", "value":' + str(in_person_contact_indicator) + '}, \
                    {"name":"linkedin_indicator", "value":' + str(linkedin_indicator) + '}, \
                    {"name":"marital_status", "value":' + str(marital_status) + '}, \
                    {"name":"num_transactions", "value":' + str(num_transactions) + '}, \
                    {"name":"primary_transfer_cat", "value":' + str(primary_transfer_cat) + '}, \
                    {"name":"prior_ctr_indicator", "value":' + str(prior_ctr_indicator) + '}, \
                    {"name":"trx_10ksum_indicator", "value":' + str(trx_10ksum_indicator) + '} \
                        ] \
            }'
    headers = {'Content-Type': 'application/vnd.sas.microanalytic.module.step.input+json', 
               'Authorization': 'Bearer ' + token}
    url = hostname_model + '/microanalyticScore/modules/' + model_name + '/steps/score'
    r = request('POST', url, data=data, headers=headers, verify=False)
    return_val = (r.json()['outputs'][2]['value'])
    return return_val

score_val = "{0:.2%}".format(score_model(1,0,1,1.508,1,1,0,0,0,0.3214,2,0,0))
print ('')
print('****************************************************************************')
print("This customer has a " + str(score_val) + " probability of a money laundering event ")
print('****************************************************************************')