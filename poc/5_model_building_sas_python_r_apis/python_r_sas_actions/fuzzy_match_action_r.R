
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
match_out_tbl <- paste(in_mem_tbl, '_match_out', sep ="")

#####################
###  Fuzzy Match  ###
#####################

loadActionSet(conn, 'dataStep')
loadActionSet(conn, 'entityRes')

cas.table.fetch(conn,
                table=list(caslib=caslib, name=in_mem_tbl)
                )

### complev generates the Levenshtein edit distance ###
### ILN ignores case, leading blanks, and quotations ###

cas.dataStep.runCode(conn,
                     code="
                          data public.credit_app_match;
                            set public.CREDIT_APPLICATION_OCR;
                            modifiers = 'ILN';
                            first_name_lev = complev(first_name, 'Christopher', modifiers);
                            first_name_rev = first_name;
                            if first_name_lev <= 2 then first_name_rev = 'Christopher';
                            last_name_lev = complev(last_name, 'Peterson', modifiers);
                            last_name_rev = last_name;
                            if last_name_lev <= 2 then last_name_rev = 'Peterson';
                          run;     
                          "
                    )

cas.table.fetch(conn,
                table=list(caslib=caslib, name=spell_dist_tbl)
                )

### match action sets rules to credit cluster_ids ###

cas.entityRes.match(conn,
                    clusterId='cluster_id', clusterIDType='CHAR',
                    columns=list('first_name','middle_name','last_name', 'email'),
                    inTable=list(caslib=caslib, name=in_mem_tbl),
                    matchRules=list(list(
                      rule=list(list(columns=list('last_name')))
                    )),
                    outTable=list(caslib=caslib, name=match_out_tbl, replace=TRUE)
                    )

cas.table.fetch(conn,
                table=list(caslib=caslib, name=match_out_tbl)
                )