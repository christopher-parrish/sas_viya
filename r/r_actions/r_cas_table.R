
###################
### Credentials ###
###################

library(sys)
library(askpass)

wd <- 'C:/...'
setwd(wd)
source(file.path(wd, 'password.r'))
username <- askpass("USERNAME")
password <- askpass("PASSWORD")
metadata_output_dir = 'outputs'

###################
### Environment ###
###################

library(swat)

### note, certificate needs to be added to 'Trusted Root Certificates' on windows client
### type in mmc to get started
### ca_cert.crt was added using these steps: https://www.thewindowsclub.com/manage-trusted-root-certificates-windows
hostname <- 'https://.../cas-shared-default-http/'
port <- 443
conn <- swat::CAS(hostname = hostname, port = port, username = username, password = password, protocol='http')
print(conn)
print(cas.builtins.serverStatus(conn))

#############################
### Identify Table in CAS ###
#############################

### caslib and table to use in scoring
caslib <- 'Public'
in_mem_tbl <- 'AML_BANK_PREP'

### load table in-memory if not already exists in-memory ###
if (cas.table.tableExists(conn, caslib=caslib, name=in_mem_tbl)<=0) {
  cas.table.loadTable(conn, caslib=caslib, path=paste(in_mem_tbl,('.sashdat')), casout=list(name=in_mem_tbl, caslib=caslib, promote=True))}

### show table to verify
cas.table.tableInfo(conn, caslib=caslib, wildIgnore=FALSE, name=in_mem_tbl)

#######################
### Run R Dataframe ###
#######################

dm_inputdf <- to.casDataFrame(defCasTable(conn, in_mem_tbl, caslib=caslib))
sapply(dm_inputdf, class)

df <- dm_inputdf[dm_inputdf['account_id'] < 5, ]
print(df)
