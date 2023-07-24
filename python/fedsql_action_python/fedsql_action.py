###################
### Credentials ###
###################

import keyring
import runpy
import os
import urllib3
urllib3.disable_warnings()

from password_poc import hostname, wd
runpy.run_path(path_name='password_poc.py')
username = keyring.get_password('cas', 'username')
password = keyring.get_password('cas', username)

###################
### Environment ###
###################

import swat

port = 443
os.environ['CAS_CLIENT_SSL_CA_LIST']=str(wd)+str('/ca_cert_poc.pem')
conn =  swat.CAS(hostname, port, username=username, password=password, protocol='http')
print(conn)

### Identify Table ###
caslib = 'Public'
in_mem_tbl = 'AML_BANK_PREP'

### load table in-memory if not already exists in-memory ###
if conn.table.tableExists(caslib=caslib, name=in_mem_tbl).exists<=0:
    conn.table.loadTable(caslib=caslib, path=str(in_mem_tbl+str('.sashdat')), 
                         casout={'name':in_mem_tbl, 'caslib':caslib, 'promote':True})
    
### show table to verify
conn.table.tableInfo(caslib=caslib, wildIgnore=False, name=in_mem_tbl)

### Show Columns in Table ###

### print columns for review of model parameters
conn.table.columnInfo(table=dict(caslib=caslib, name=in_mem_tbl))

### Run SAS Query ###

conn.loadactionset('fedSql')

df = conn.fedSql.execDirect(
    query='''
    
    select * from public.aml_bank_prep
    where account_id <5;
    
    ''')

display(df)
