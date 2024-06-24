
####################################################
###      Score Model Deployed to MAS by API      ###
####################################################

###################
### Credentials ###
###################

import os
import sys
from pathlib import Path

sys.path.append('C:/Users/chparr/OneDrive - SAS/credentials')
from credentials import hostname, session, port, protocol, wd, output_dir, git_dir, token_dir, token, token_refresh, token_pem, username


###################
### Environment ###
###################

import swat

access_token = open(token, "r").read()
conn =  swat.CAS(hostname=hostname, username=None, password=access_token, ssl_ca_list=token_pem, protocol=protocol)
print(conn.serverstatus())

#################
### Get Token ###
#################

from requests import request

# url = session + '/SASLogon/oauth/token' 
# r = request('POST', url,
#             data='grant_type=password&username=%s&password=%s' %(username, password),
#             headers={
#                 'Accept': 'application/json',
#                 'Content-Type': 'application/x-www-form-urlencoded'
#             },
#             auth=('sas.ec', ''),
#             verify=False)
# access_token = r.json()['access_token']

############################
### View Deployed Models ###
############################

headers = {'Authorization': 'Bearer ' + access_token}
url = session + '/microanalyticScore/modules/'
r = request('GET', url, params={}, headers=headers, verify=False)
for key in r.json()['items']:
    print(key['name'].lower())

# the list may be incomplete #

###########################
### Select Target Model ###
###########################

#model_name = 'logit_sas_iml_amlbank_sasstudio' # all lower case
model_name = 'logit_python_aml_bank' # all lower case

headers = {'Authorization': 'Bearer ' + access_token}
url = session + '/microanalyticScore/modules/' + model_name + '/steps'
r = request('GET', url, params={}, headers=headers, verify=False)
r.json()

# note: the output contains the 'inputs' and 'outputs' metadata associated with the scoring #

##########################
### Score Target Model ###
##########################

data = '{"inputs":[ {"name":"marital_status_single", "value": 1}, \
                    {"name":"checking_only_indicator", "value": 1}, \
                    {"name":"prior_ctr_indicator", "value": 1}, \
                    {"name":"address_change_2x_indicator", "value": 1}, \
                    {"name":"cross_border_trx_indicator", "value": 1}, \
                    {"name":"in_person_contact_indicator", "value": 1}, \
                    {"name":"linkedin_indicator", "value": 1}, \
                    {"name":"citizenship_country_risk", "value": 4}, \
                    {"name":"distance_to_employer", "value": 0.04}, \
                    {"name":"distance_to_bank", "value": 0.005} \
                        ] \
            }'
headers = {'Content-Type': 'application/vnd.sas.microanalytic.module.step.input+json', 
               'Authorization': 'Bearer ' + access_token}
url = session + '/microanalyticScore/modules/' + model_name + '/steps/score'
r = request('POST', url, data=data, headers=headers, verify=False)
score_val = (r.json()['outputs'][2]['value'])
score_val = "{0:.2%}".format(score_val)


#print ('')
#print('****************************************************************************')
print("This customer has a " + str(score_val) + " probability of a money laundering event ")
#print('****************************************************************************')

# as score function #

headers = {'Content-Type': 'application/vnd.sas.microanalytic.module.step.input+json', 
               'Authorization': 'Bearer ' + access_token}
url = session + '/microanalyticScore/modules/' + model_name + '/steps/score'
r = request('POST', url, data=data, headers=headers, verify=False)
r.json()

def score_model(marital_status_single, checking_only_indicator, prior_ctr_indicator, address_change_2x_indicator, cross_border_trx_indicator, in_person_contact_indicator, linkedin_indicator, citizenship_country_risk, distance_to_employer, distance_to_bank):
    data = '{"inputs":[ {"name":"marital_status_single", "value":' + str(marital_status_single) + '}, \
                    {"name":"checking_only_indicator", "value":' + str(checking_only_indicator) + '}, \
                    {"name":"prior_ctr_indicator", "value":' + str(prior_ctr_indicator) + '}, \
                    {"name":"address_change_2x_indicator", "value":' + str(address_change_2x_indicator) + '}, \
                    {"name":"cross_border_trx_indicator", "value":' + str(cross_border_trx_indicator) + '}, \
                    {"name":"in_person_contact_indicator", "value":' + str(in_person_contact_indicator) + '}, \
                    {"name":"linkedin_indicator", "value":' + str(linkedin_indicator) + '}, \
                    {"name":"citizenship_country_risk", "value":' + str(citizenship_country_risk) + '}, \
                    {"name":"distance_to_employer", "value":' + str(distance_to_employer) + '}, \
                    {"name":"distance_to_bank", "value":' + str(distance_to_bank) + '} \
                        ] \
            }'
    headers = {'Content-Type': 'application/vnd.sas.microanalytic.module.step.input+json', 
               'Authorization': 'Bearer ' + access_token}
    url = session + '/microanalyticScore/modules/' + model_name + '/steps/score'
    r = request('POST', url, data=data, headers=headers, verify=False)
    return_val = (r.json()['outputs'][2]['value'])
    return return_val

score_val = "{0:.2%}".format(score_model(1,1,1,1,1,1,1,0.04,0.005))


print ('')
print('****************************************************************************')
print("This customer has a " + str(score_val) + " probability of a money laundering event ")
print('****************************************************************************')