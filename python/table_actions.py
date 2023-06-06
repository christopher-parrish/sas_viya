####################################################
###  Train & Register Python Scikit Logit Model  ###
####################################################

###################
### Credentials ###
###################

import keyring
import runpy
import os

### run script that contains username, password, hostname, working directory, and output directory
    ### ...OR define directly in this script
from password import hostname, port, wd, output_dir, hostname_dev, port_dev

#runpy.run_path(path_name='password.py')
username = keyring.get_password('cas', 'username')
password = keyring.get_password('cas', username)
metadata_output_dir = 'outputs'

###################
### Environment ###
###################

import swat
import pandas as pd

#os.environ['CAS_CLIENT_SSL_CA_LIST'] = "C:\\Users\\chparr\\OneDrive - SAS\\sas_viya_cl_executable\\trustedcerts.pem"
#conn = swat.CAS(hostname_dev, port_dev, username, password, protocol='https')

conn =  swat.CAS(hostname, port, username=username, password=password, protocol='cas')
print(conn.serverstatus())

#############################
### Identify Table in CAS ###
#############################

### caslib and table to use in modeling
caslib = 'Public'
in_mem_tbl = 'FINANCIAL_SERVICES_PREP'

### load table in-memory if not already exists in-memory
if conn.table.tableExists(caslib=caslib, name=in_mem_tbl).exists<=0:
    conn.table.loadTable(caslib=caslib, path=str(in_mem_tbl+str('.sashdat')), 
                         casout={'name':in_mem_tbl, 'caslib':caslib, 'promote':True})

### show table to verify
conn.table.tableInfo(caslib=caslib, wildIgnore=False, name=in_mem_tbl)

########################
### Create Dataframe ###
########################

dm_inputdf =  conn.CASTable(in_mem_tbl, caslib=caslib).to_frame()

### print columns for review of model parameters
print(dm_inputdf.dtypes)

########################
### Upload Dataframe ###
########################

dm_inputdf.to_csv(str(wd)+str('/')+in_mem_tbl+str('.csv'))

conn.upload(str(wd)+str('/')+in_mem_tbl+str('.csv'))



### read csv from defined 'data_dir' directory
#data_dir = 'C:/'
#dm_inputdf = pd.read_csv(str(data_dir)+str('/')+in_mem_tbl+str('.csv'))



