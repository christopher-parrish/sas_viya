#%%
###################
### Credentials ###
###################

import keyring
import runpy
import os

### run script that contains username, password, hostname, and working directory
    ### ...OR define directly in this script
from password_poc import hostname, output_dir, wd
runpy.run_path(os.path.join(wd, 'password_poc.py'))
username = keyring.get_password('cas', 'username')
password = keyring.get_password('cas', username)

###################
### Environment ###
###################

import swat

port = 443
cert_path = str(wd)+str('/ca_cert_poc.pem')
os.environ['CAS_CLIENT_SSL_CA_LIST']=cert_path
conn =  swat.CAS(hostname, port, username=username, password=password, protocol='http')
#print(conn)
#print(conn.serverstatus())

### caslib and table to use in scoring
caslib = 'Public'
in_mem_tbl = 'black_scholes_score'

########################
### Model Parameters ###
########################

import pandas as pd
import shutil
from pathlib import Path
import sys
from datetime import datetime

### model manager information
model_name = 'DEMO_black_scholes_python'
score_code_name = 'DEMO_black_scholes_pythonScore.py'
project_name = 'Black_Scholes_Option'
description = 'UDF'
model_type = 'UDF'
metadata_output_dir = 'outputs'
dm_dec_target = 'option_val'
python_version = sys.version
timestamp = str(datetime.now())
#%%
#########################
### Create json Files ###
#########################

import json

inputVar = [
    {
        "name": "notional",
        "length": 8,
        "type": "decimal",
        "level": "interval"
    },
    {
        "name": "vol",
        "length": 8,
        "type": "decimal",
        "level": "interval"
    },
    {
        "name": "strike_price",
        "length": 8,
        "type": "decimal",
        "level": "interval"
    },
    {
        "name": "spot_price",
        "length": 8,
        "type": "decimal",
        "level": "interval"
    },
    {
        "name": "time_to_mat",
        "length": 8,
        "type": "decimal",
        "level": "interval"
    },
    {
        "name": "risk_free_rate",
        "length": 8,
        "type": "decimal",
        "level": "interval"
    }
]


outputVar = [
    {
        "name": "option_val",
        "length": 8,
        "type": "decimal",
        "level": "interval"
    }
]


fileMetadata = [
    {
        "role": "inputVariables",
        "name": "inputVar.json"
    },
    {
        "role": "outputVariables",
        "name": "outputVar.json"
    },
    {
        "role": "score",
        "name": score_code_name
    },
    {
        "role": "scoreResource",
        "name": "requirements.json"
    }
]


ModelProperties = {
  "custom properties": [],
  "externalUrl": "",
  "predictionVariable": "",
  "modelVersionName": "",
  "trainTable": "",
  "trainCodeType": "python",
  "description": description,
  "tool": "Python 3",
  "toolVersion": "3.8.8",
  "targetVariable": "",
  "scoreCodeType": "python",
  "externalModelId": "",
  "createdBy": username,
  "function": "prediction",
  "eventProbVar": dm_dec_target,
  "modeler": username,
  "name": model_name,
  "modifiedTimeStamp": "",
  "modifiedBy": username,
  "id": "",
  "creationTimeStamp": "",
  "targetEvent": "",
  "targetLevel": "",
  "algorithm": model_type
}

requirements = [
    {
        "step":"import numpy as np",
        "command":"pip3 install numpy==1.20.3"
    },
    {
        "step":"from scipy.stats import norm",
        "command":"pip3 install scipy==1.7.1"
    }
]

output_path = Path(output_dir) / metadata_output_dir / model_name
if output_path.exists() and output_path.is_dir():
    shutil.rmtree(output_path)

os.makedirs(output_path)

inputVarObj = json.dumps(inputVar, indent = 4)
with open(str(output_path)+str('/inputVar.json'), 'w') as outfile:
    outfile.write(inputVarObj)

outputVarObj = json.dumps(outputVar, indent = 4)
with open(str(output_path)+str('/outputVar.json'), 'w') as outfile:
    outfile.write(outputVarObj)

fileMetadataObj = json.dumps(fileMetadata, indent = 4)
with open(str(output_path)+str('/fileMetadata.json'), 'w') as outfile:
    outfile.write(fileMetadataObj)
    
ModelPropertiesObj = json.dumps(ModelProperties, indent = 4)
with open(str(output_path)+str('/ModelProperties.json'), 'w') as outfile:
    outfile.write(ModelPropertiesObj)

requirementsObj = json.dumps(requirements, indent = 4)
with open(str(output_path)+str('/requirements.json'), 'w') as outfile:
    outfile.write(requirementsObj)

#########################
### Create Score Code ###
#########################

score_script = """
    
import numpy as np
from scipy.stats import norm

def option_value(notional, vol, strike_price, spot_price, time_to_mat, risk_free_rate):
    "Output: option_val"
    d1 = (np.log(spot_price/strike_price) + (risk_free_rate+vol**2/2)*time_to_mat)/(vol*np.sqrt(time_to_mat))
    d2 = d1 - vol*np.sqrt(time_to_mat)
    option_price = norm.cdf(d1)*spot_price-norm.cdf(d2)*strike_price*np.exp(-risk_free_rate*time_to_mat)
    option_val = notional * option_price
    return float(option_val)
"""
    
with open((output_path / score_code_name), 'w') as scorecode:
    scorecode.write(score_script)
                    
###################################
### Create Table to Score Model ###
###################################

data_x = [[200, .5, 15, 5, 1, 1]]
columns_x = ['notional', 'vol', 'strike_price', 'spot_price', 'time_to_mat', 'risk_free_rate']
X = pd.DataFrame(data_x, columns=columns_x)
if conn.table.tableExists(caslib=caslib, name=in_mem_tbl).exists<=0:
    conn.upload(data=X, casOut={"caslib":caslib, "name":in_mem_tbl, "promote":True})
#%%
#########################################
### Zip Files & Send to Model Manager ###
#########################################

from sasctl import Session
import sasctl.pzmm as pzmm
from sasctl._services.model_repository import ModelRepository as mr

zip_file = pzmm.ZipModel.zipFiles(fileDir=output_path, modelPrefix=model_name, isViya4=True)
sess=Session(hostname, username=username, password=password, verify_ssl=False, protocol="http")
with sess:
    try:
        mr.get_project(project_name).name
    except:
        mr.create_project(project_name, mr.default_repository())
    mr.import_model_from_zip(model_name, project_name, zip_file, version='latest')
