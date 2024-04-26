###################
### Credentials ###
###################


###################
### Environment ###
###################


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

#######################################
### Create Dataframe from CAS Table ###
#######################################

dm_inputdf =  conn.CASTable(in_mem_tbl, caslib=caslib).to_frame()

### print columns for review of model parameters
print(dm_inputdf.dtypes)

##############################
### Write Dataframe to CSV ###
##############################

dm_inputdf.to_csv(str(wd)+str('/')+in_mem_tbl+str('.csv'))

##############################
### Upload CSV to SAS Viya ###
##############################

conn.upload(str(wd)+str('/')+in_mem_tbl+str('.csv'))

########################
### Read CSV Locally ###
########################

### read csv from defined 'data_dir' directory
data_dir = 'C:/'
dm_inputdf = pd.read_csv(str(data_dir)+str('/')+in_mem_tbl+str('.csv'))



