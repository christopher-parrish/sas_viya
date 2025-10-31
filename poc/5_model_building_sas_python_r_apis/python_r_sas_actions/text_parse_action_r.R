
########################################################
###  Optical Character Recognition (OCR) Extraction  ###
########################################################

###################
### Credentials ###
###################

library(askpass)
library(sys)

username <- askpass("USERNAME")
password <- askpass("PASSWORD")
wd <- askpass("What is the Working Directory for this R Session?")
source(file.path(wd, 'password.r'))

###################
### Environment ###
###################

library(swat)

conn <- swat::CAS(hostname=hostname, port=port, username, password, protocol=protocol)
print(cas.builtins.serverStatus(conn))

#############################
### Identify Table in CAS ###
#############################

### caslib and table to use in modeling
caslib <- 'Public'
in_mem_tbl <- 'CREDIT_APPLICATION_OCR'

### load table in-memory if not already exists in-memory
if (cas.table.tableExists(conn, caslib=caslib, name=in_mem_tbl)<=0) {
  cas.table.loadTable(conn, caslib=caslib, path=paste(in_mem_tbl,('.sashdat'), sep = ""), 
                      casout=list(name=in_mem_tbl, caslib=caslib, promote=TRUE))}

### show table to verify
cas.table.tableInfo(conn, caslib=caslib, wildIgnore=FALSE, name=in_mem_tbl)

### create names of tables for action set
parse_out_tbl <- paste(in_mem_tbl, '_parse_out', sep ="")
spell_out_tbl <- paste(in_mem_tbl, '_spell_out', sep ="")
spell_dist_tbl <- 'credit_app_match'

### this is incomplete ###

####################
###  Parse Text  ###
####################

loadActionSet(conn, 'textParse')

cas.textParse.tpParse(conn,
                      docId='_Index_',
                      offset=list(caslib=caslib, name=parse_out_tbl, replace=TRUE),
                      table=list(caslib=caslib, name=in_mem_tbl),
                      text="first_name"   
                      )

cas.textParse.tpSpell(conn,
                      casOut=list(caslib=caslib, name=spell_out_tbl, replace=TRUE),
                      table=list(caslib=caslib, name=in_mem_tbl)
                      )

cas.table.fetch(conn,
                table=list(caslib=caslib, name=spell_out_tbl)
                )