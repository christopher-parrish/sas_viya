
############################################################
###    Score Decision Deployed to a Container by API     ###
############################################################

from requests import request
import http
import json
import pandas as pd

###################################
###    Connect to Container     ###
###################################

# SCR listens on port 8080
# Use the containers fully qualified domain name (FQDN)
# Ensure the container instance is set up as 'public' (no credentials required)
# URL is the name of the container image to be used
# The container needs to be running when the code is executed
# Helpful video: https://communities.sas.com/t5/SAS-Communities-Library/How-to-Score-a-SAS-Decision-Published-to-Azure-with-SCR/ta-p/777629

conn = http.client.HTTPConnection("financialservicesdecision.eastus.azurecontainer.io:8080")
url = "/financial_services_decision"

####################################
### Score Decision as a Function ###
####################################

def score_container(age, amount, at_current_job_1_year, business_owner, citizenship_africa, citizenship_china, citizenship_eu, citizenship_india, citizenship_latam,
                citizenship_middle_east, citizenship_other, citizenship_us, credit_history_mos, credit_score, debt_to_income, ever_missed_obligation, gender,
                 homeowner, id_current_fs_relationship, id_direct_contact, id_important_activity, job_in_education, job_in_energy, job_in_financial, 
                  job_in_healthcare, job_in_hospitality, job_in_manufacturing, job_in_media, job_in_transport, martial_status_divorced, marital_status_married, 
                   marital_status_single, net_worth, num_dependents, num_transactions, orig_age, orig_amount, orig_credit_history_mos, orig_credit_score,
                    orig_debt_to_income, orig_net_worth, orig_num_transactions, orig_years_at_residence, region_ca, region_fl, region_mw, region_ne, region_ny,
                     region_so, region_tx, region_we, smoker, uses_direct_deposit, years_at_residence):
    payload = '{"inputs":[ {"name":"age", "value":' + str(age) + '}, \
                    {"name":"amount", "value":' + str(amount) + '}, \
                    {"name":"at_current_job_1_year", "value":' + str(at_current_job_1_year) + '}, \
                    {"name":"business_owner", "value":' + str(business_owner) + '}, \
                    {"name":"citizenship_africa", "value":' + str(citizenship_africa) + '}, \
                    {"name":"citizenship_china", "value":' + str(citizenship_china) + '}, \
                    {"name":"citizenship_eu", "value":' + str(citizenship_eu) + '}, \
                    {"name":"citizenship_india", "value":' + str(citizenship_india) + '}, \
                    {"name":"citizenship_latam", "value":' + str(citizenship_latam) + '}, \
                    {"name":"citizenship_middle_east", "value":' + str(citizenship_middle_east) + '}, \
                    {"name":"citizenship_other", "value":' + str(citizenship_other) + '}, \
                    {"name":"citizenship_us", "value":' + str(citizenship_us) + '}, \
                    {"name":"credit_history_mos", "value":' + str(credit_history_mos) + '}, \
                    {"name":"credit_score", "value":' + str(credit_score) + '}, \
                    {"name":"debt_to_income", "value":' + str(debt_to_income) + '}, \
                    {"name":"ever_missed_obligation", "value":' + str(ever_missed_obligation) + '}, \
                    {"name":"gender", "value":' + str(gender) + '}, \
                    {"name":"homeowner", "value":' + str(homeowner) + '}, \
                    {"name":"id_current_fs_relationship", "value":"' + id_current_fs_relationship + '"}, \
                    {"name":"id_direct_contact", "value":"' + id_direct_contact + '"}, \
                    {"name":"id_important_activity", "value":"' + id_important_activity + '"}, \
                    {"name":"job_in_education", "value":' + str(job_in_education) + '}, \
                    {"name":"job_in_energy", "value":' + str(job_in_energy) + '}, \
                    {"name":"job_in_financial", "value":' + str(job_in_financial) + '}, \
                    {"name":"job_in_healthcare", "value":' + str(job_in_healthcare) + '}, \
                    {"name":"job_in_hospitality", "value":' + str(job_in_hospitality) + '}, \
                    {"name":"job_in_manufacturing", "value":' + str(job_in_manufacturing) + '}, \
                    {"name":"job_in_media", "value":' + str(job_in_media) + '}, \
                    {"name":"job_in_transport", "value":' + str(job_in_transport) + '}, \
                    {"name":"martial_status_divorced", "value":' + str(martial_status_divorced) + '}, \
                    {"name":"marital_status_married", "value":' + str(marital_status_married) + '}, \
                    {"name":"marital_status_single", "value":' + str(marital_status_single) + '}, \
                    {"name":"net_worth", "value":' + str(net_worth) + '}, \
                    {"name":"num_dependents", "value":' + str(num_dependents) + '}, \
                    {"name":"num_transactions", "value":' + str(num_transactions) + '}, \
                    {"name":"orig_age", "value":' + str(orig_age) + '}, \
                    {"name":"orig_amount", "value":' + str(orig_amount) + '}, \
                    {"name":"orig_credit_history_mos", "value":' + str(orig_credit_history_mos) + '}, \
                    {"name":"orig_credit_score", "value":' + str(orig_credit_score) + '}, \
                    {"name":"orig_debt_to_income", "value":' + str(orig_debt_to_income) + '}, \
                    {"name":"orig_net_worth", "value":' + str(orig_net_worth) + '}, \
                    {"name":"orig_num_transactions", "value":' + str(orig_num_transactions) + '}, \
                    {"name":"orig_years_at_residence", "value":' + str(orig_years_at_residence) + '}, \
                    {"name":"region_ca", "value":' + str(region_ca) + '}, \
                    {"name":"region_fl", "value":' + str(region_fl) + '}, \
                    {"name":"region_mw", "value":' + str(region_mw) + '}, \
                    {"name":"region_ne", "value":' + str(region_ne) + '}, \
                    {"name":"region_ny", "value":' + str(region_ny) + '}, \
                    {"name":"region_so", "value":' + str(region_so) + '}, \
                    {"name":"region_tx", "value":' + str(region_tx) + '}, \
                    {"name":"region_we", "value":' + str(region_we) + '},\
                    {"name":"smoker", "value":' + str(smoker) + '}, \
                    {"name":"uses_direct_deposit", "value":' + str(uses_direct_deposit) + '}, \
                    {"name":"years_at_residence", "value":' + str(years_at_residence) + '} \
                        ] \
            }'

    headers = {'Content-Type': 'application/json', 
               'Content-Type': 'application/json'}
    conn.request('POST', url, payload, headers)
    response = conn.getresponse()
    return_values = response.read()
    json_load = json.loads(return_values)
    json_data = json_load['outputs']
    return json_data

json_data = score_container(0.4235,1.5145,1,1,0,0,0,0,0,0,0,1,-1.9745,0.3043,-1.6773,0,0,1,"false","false","false",0,0,0,0,0,0,1,0,0,0,1,-0.7583,2,1.9319,49,325888,23,729,5,108135,74,26,0,0,1,0,0,0,0,0,0,0,1.8208)

df = pd.DataFrame.from_dict(pd.json_normalize(json_data), orient='columns')
df = df.reset_index(drop=True)
approveFlag = df.loc[df['name']=="ApproveFlag"]
declineFlag = df.loc[df['name']=="DeclineFlag"]
declineReason = df.loc[df['name']=="DeclineReason"]
loanDecision = df.loc[df['name']=="loan_decision"]
print(approveFlag['value'])
print(declineFlag['value'])
print(declineReason['value'])
print(loanDecision['value'])

print ('')
print('****************************************************************************')
print("Your loan decision is " + str(loanDecision['value'].to_string(index=False)))
print("If denied, the reason for denial is " + str(declineReason['value'].to_string(index=False)))
print('****************************************************************************')


###################################
### Score Decision Individually ###
###################################

payload = json.dumps({
    "inputs":[
    {"name":"age","value":0.4235},
	{"name":"amount","value":1.5145},
	{"name":"at_current_job_1_year","value":1},
	{"name":"business_owner","value":1},
	{"name":"citizenship_africa","value":0},
	{"name":"citizenship_china","value":0},
	{"name":"citizenship_eu","value":0},
	{"name":"citizenship_india","value":0},
	{"name":"citizenship_latam","value":0},
	{"name":"citizenship_middle_east","value":0},
    {"name":"citizenship_other","value":0},
    {"name":"citizenship_us","value":1},
    {"name":"credit_history_mos","value":-1.9745},
    {"name":"credit_score","value":0.3043},
    {"name":"debt_to_income","value":-1.6773},
    {"name":"event_indicator","value":0},
    {"name":"ever_missed_obligation","value":0},
    {"name":"gender","value":0},
    {"name":"homeowner","value":1},
    {"name":"id_current_fs_relationship","value":"false"},
    {"name":"id_direct_contact","value":"false"},
    {"name":"id_important_activity","value":"false"},
    {"name":"job_in_education","value":0},
    {"name":"job_in_energy","value":0},
    {"name":"job_in_financial","value":0},
    {"name":"job_in_healthcare","value":0},
    {"name":"job_in_hospitality","value":0},
    {"name":"job_in_manufacturing","value":0},
    {"name":"job_in_media","value":1},
    {"name":"job_in_transport","value":0},
    {"name":"marital_status_divorced","value":0},
    {"name":"marital_status_married","value":0},
    {"name":"marital_status_single","value":1},
    {"name":"net_worth","value":-0.7583},
    {"name":"num_dependents","value":2},
    {"name":"num_transactions","value":1.9319},
    {"name":"orig_age","value":49},
    {"name":"orig_amount","value":325888},
    {"name":"orig_credit_history_mos","value":23},
    {"name":"orig_credit_score","value":729},
    {"name":"orig_debt_to_income","value":5},
    {"name":"orig_net_worth","value":108135},
    {"name":"orig_num_transactions","value":74},
    {"name":"orig_years_at_residence","value":26},
    {"name":"region_ca","value":0},
    {"name":"region_fl","value":0},
    {"name":"region_mw","value":1},
    {"name":"region_ne","value":0},
    {"name":"region_ny","value":0},
    {"name":"region_so","value":0},
    {"name":"region_tx","value":0},
    {"name":"region_we","value":0},
    {"name":"smoker","value":0},
    {"name":"uses_direct_deposit","value":0},
    {"name":"years_at_residence","value":1.8208}
    ]
    })

headers = {'Content-Type': 'application/json', 
               'Content-Type': 'application/json'}
conn.request('POST', url, payload, headers)
response = conn.getresponse()
return_values = response.read()
json_load = json.loads(return_values)
json_data = json_load['outputs']

df = pd.DataFrame.from_dict(pd.json_normalize(json_data), orient='columns')
df = df.reset_index(drop=True)
approveFlag = df.loc[df['name']=="ApproveFlag"]
declineFlag = df.loc[df['name']=="DeclineFlag"]
declineReason = df.loc[df['name']=="DeclineReason"]
loanDecision = df.loc[df['name']=="loan_decision"]
print(approveFlag['value'])
print(declineFlag['value'])
print(declineReason['value'])
print(loanDecision['value'])

print ('')
print('****************************************************************************')
print("Your loan decision is " + str(loanDecision['value'].to_string(index=False)))
print("If denied, the reason for denial is " + str(declineReason['value'].to_string(index=False)))
print('****************************************************************************')
