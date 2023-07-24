#######################################
###      Create an ARIMA Model      ###
#######################################

###################
### Credentials ###
###################

import keyring
import getpass
import runpy
import os
from pathlib import Path
import urllib3
urllib3.disable_warnings()

### run script that contains username, password, hostname, working directory, and output directory
    ### ...OR define directly in this script
from password import hostname, port, wd, output_dir
runpy.run_path(path_name='password.py')
username = keyring.get_password('cas', 'username')
password = keyring.get_password('cas', username)
# username = getpass.getpass("Username: ")
# password = getpass.getpass("Password: ")
output_dir = os.getcwd()
metadata_output_dir = 'outputs'

###################
### Environment ###
###################

import swat
import pandas as pd

conn = swat.CAS(hostname, port, username, password, protocol="cas")
print(conn)
print(conn.serverstatus())

#############################
### Identify Table in CAS ###
#############################

### caslib and table to use in modeling
caslib = 'Public'
in_mem_tbl = 'AIR'

### load table in-memory if not already exists in-memory
if conn.table.tableExists(caslib=caslib, name=in_mem_tbl).exists<=0:
    conn.table.loadTable(caslib=caslib, path=str(in_mem_tbl+str('.sashdat')), 
                         casout={'name':in_mem_tbl, 'caslib':caslib, 'promote':True})
    
### show table to verify
conn.table.tableInfo(caslib=caslib, wildIgnore=False, name=in_mem_tbl)

########################
### Create Dataframe ###
########################

dm_inputdf =  conn.CASTable(in_mem_tbl, caslib=caslib)

### print columns for review of model parameters
conn.table.columnInfo(table={"caslib":caslib, "name":in_mem_tbl})

########################
### Model Parameters ###
########################

### import packages
conn.loadactionset('uniTimeSeries')

conn.uniTimeSeries.arima(
      table=dict(caslib=caslib, name=in_mem_tbl),
      timeId=dict(name='DATE'),
      interval='MONTH',
      outEst=dict(caslib=caslib, name='ARIEST', replace=True),
      outFor=dict(caslib=caslib, name='ARIFOR', replace=True),
      series=[(dict(name='AIR',
                       model=[
                             dict(
                                 estimate=dict(
                                                p=dict(factor=1),
                                                method='ML', 
                                                diff=1,
                                                transform='log'
                                               ),
                                forecast=dict(lead=6)
                                 )
]))])

conn.table.fetch(
      table=dict(caslib=caslib, name='ARIFOR',
                 where='date>=''01SEP1960''d')
                )

