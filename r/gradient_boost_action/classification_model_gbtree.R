
#####################################################
###       Train & Publish SAS gbtree Model        ###
#####################################################

###################
### Credentials ###
###################

library(askpass)
library(sys)

#setwd("C:..") # or set in Session tab
wd <- getwd()
source(file.path(wd, 'password.r'))
username <- askpass("USERNAME")
password <- askpass("PASSWORD")
output_dir <- getwd()
metadata_output_dir <- 'outputs'

###################
### Environment ###
###################

library(swat)

hostname <- 'https://fsbulab.unx.sas.com/cas-shared-default-http/'
port <- 443
conn <- swat::CAS(hostname, port, username, password, protocol='http')
print(cas.builtins.serverStatus(conn))

#############################
### Identify Table in CAS ###
#############################

### caslib and table to use in modeling
caslib <- 'Public'
in_mem_tbl <- 'AML_BANK_PREP'

### load table in-memory if not already exists in-memory
if (cas.table.tableExists(conn, caslib=caslib, name=in_mem_tbl)<=0) {
  cas.table.loadTable(conn, caslib=caslib, path=paste(in_mem_tbl,('.sashdat')), 
                      casout=list(name=in_mem_tbl, caslib=caslib, promote=True))}

### show table to verify
cas.table.tableInfo(conn, caslib=caslib, wildIgnore=FALSE, name=in_mem_tbl)

### create names of tables for action set
astore_tbl <- paste(in_mem_tbl, '_astore', sep ="")
cas_score_tbl <- paste(in_mem_tbl, '_score', sep ="")
cas_out_tbl <- paste(in_mem_tbl, '_model', sep ="")

########################
### Create Dataframe ###
########################

dm_inputdf <- defCasTable(conn, in_mem_tbl, caslib=caslib)

### print columns for review of model parameters
cas.table.columnInfo(conn, table=list(caslib=caslib, name=in_mem_tbl))

########################
### Model Parameters ###
########################

### import packages
loadActionSet(conn, 'decisionTree')
loadActionSet(conn, 'astore')
loadActionSet(conn, 'explainModel')
loadActionSet(conn, 'fairAITools')
loadActionSet(conn, 'percentile')
loadActionSet(conn, 'modelPublishing')

### model arguments
m=20
seed=12345
nTree=100
learningRate=0.1
subSampleRate=0.5
lasso=0
ridge=1
distribution="binary"
maxBranch=2
maxLevel=5
leafSize=5
missing="useinsearch"
minUseInSearch=1
nBins=50
quantileBin=TRUE

early_stop_params = list(
  metric="MCR",
  stagnation=5,
  tolerance=0,
  minimum=FALSE,
  threshold=0,
  thresholdIter=0
)

### model manager information
model_name <- 'gbtree_r'
project_name <- 'Risk Score'
description <- 'gbtree_r'
model_type <- 'gradient_boost'

### define macro variables for model
dm_dec_target <- 'ml_indicator'
dm_partitionvar <- 'analytic_partition'
create_new_partition <- 'no' # 'yes'/'no'
dm_key <- 'account_id' 
dm_classtarget_level <- list('0', '1')
dm_partition_val <- list('dm_partition_validate_val'=0, 'dm_partition_train_val'=1, 'dm_partition_test_val'=2)
dm_partition_perc <- list('dm_partition_validate_perc'=0.3, 'dm_partition_train_perc'=0.6, 'dm_partition_test_perc'=0.1)

### create list of rejected predictor columns
rejected_predictors <- list(
  'atm_deposit_indicator', 
  'citizenship_country_risk', 
  'distance_to_bank',
  'distance_to_employer', 
  'income', 
  'num_acctbal_chgs_gt2000',
  'occupation_risk')

### var to consider in bias assessment
bias_var <- 'cross_border_trx_indicator'

##############################
### Final Modeling Columns ###
##############################

### create list of model variables
dm_input <- colnames(dm_inputdf)
macro_vars <- list(dm_dec_target, dm_partitionvar, dm_key)
rejected_vars <- unlist(c(rejected_predictors, macro_vars))
dm_input <- dm_input[! dm_input %in% c(rejected_vars)]
  
### create prediction variables
dm_predictionvar <- list((paste("P_", dm_dec_target, dm_classtarget_level[1], sep ="")),
                         (paste("P_", dm_dec_target, dm_classtarget_level[2], sep ="")),
                         'EM_EVENTPROBABILITY')
dm_classtarget_intovar <- list((paste("I_", dm_dec_target, sep ="")), 'EM_CLASSIFICATION')

### create partition objects
train_part = paste(dm_partitionvar, '=', dm_partition_val[['dm_partition_train_val']], sep="")
test_part = paste(dm_partitionvar, '=', dm_partition_val[['dm_partition_test_val']], sep="")
valid_part = paste(dm_partitionvar, '=', dm_partition_val[['dm_partition_validate_val']], sep="")

#####################
### Training Code ###
#####################

### estimate & fit model
dm_model <- cas.decisionTree.gbtreeTrain(conn,
    earlyStop=early_stop_params,
    table=list(caslib=caslib, name=in_mem_tbl, where=train_part),
    target=dm_dec_target,
    nominal=dm_dec_target,
    inputs=dm_input,
    encodeName=TRUE,
    casOut=list(caslib=caslib, name=cas_out_tbl, replace=TRUE),
    saveState=list(caslib=caslib, name=astore_tbl, replace=TRUE),
    m=m, seed=seed, nTree=nTree, learningRate=learningRate, subSampleRate=subSampleRate, 
    lasso=lasso, ridge=ridge, distribution=distribution, maxBranch=maxBranch, 
    maxLevel=maxLevel, leafSize=leafSize, missing=missing, minUseInSearch=minUseInSearch, 
    nBins=nBins, quantileBin=quantileBin 
    )

### create score code
cas.decisionTree.gbtreeCode(conn,
  modelTable=list(caslib=caslib, name=cas_out_tbl),
  code=list(casOut=list(caslib=caslib, name='gbtree_scorecode', replace=TRUE, promote=FALSE))
  )

### score full data
cas.decisionTree.dtreeScore(conn,
  modelTable=list(caslib=caslib, name=cas_out_tbl),
  table=list(caslib=caslib, name=in_mem_tbl), 
  copyvars=list(dm_dec_target, dm_partitionvar),
  casout=list(caslib=caslib, name=cas_score_tbl, replace=TRUE),
  encodeName=TRUE,
  assessOneRow=TRUE
  )

####################
### Assess Model ###
####################

cas.percentile.assess(conn,
  table=list(caslib=caslib, name=cas_score_tbl),
  event="1",
  response=dm_dec_target,
  inputs=dm_predictionvar[2],
  cutStep=0.0001,
  casOut=list(caslib=caslib, name='gbtree_r_assess', replace=TRUE)
  )

###################
### Assess Bias ###
###################

cas.fairAITools.assessBias(conn,
  table = list(caslib=caslib, name=in_mem_tbl),
  modelTable = list(caslib=caslib, name=astore_tbl),
  modelTableType = "ASTORE",
  response = dm_dec_target,
  predictedVariables = list(dm_predictionvar[[1]], dm_predictionvar[[2]]),
  responseLevels = dm_classtarget_level,
  sensitiveVariable = bias_var
)

#################################
### Publish Model in SAS Viya ###
#################################

cas.modelPublishing.publishModel(conn, 
  modelName=model_name,
  modelTable=list(caslib=caslib, name='published_table', persist=TRUE, replace=TRUE),
  storeTables=list(list(caslib=caslib, name=astore_tbl)),
  modelType="DS2"
  )

#############################
### Run Model in SAS Viya ###
#############################

cas.modelPublishing.runModelLocal(conn, 
  modelName=model_name,
  modelTable=list(caslib=caslib, name='published_table'),
  inTable=list(caslib=caslib, name=in_mem_tbl),
  outTable=list(caslib=caslib, name='published_table_score')
  )

#######################################################
### Plot ROC & Retrieve Stats Using ggplot & Base R ###
#######################################################

library(ggplot2)
roc_cas_df <- to.casDataFrame(defCasTable(conn, caslib=caslib, table='gbtree_r_assess_ROC'))
roc_r_df <- data.frame(roc_cas_df)
line <- ggplot(data=roc_r_df, aes(x=X_FPR_, y=X_Sensitivity_)) + geom_line()
line+ggtitle("gbtree ROC")
stats <- subset(roc_r_df, roc_r_df['X_KS_']==1)
print(paste("ks=", stats['X_KS2_']), quote=FALSE)
print(paste("auc=", stats['X_C_']), quote=FALSE)
print(paste("cutoff=", stats['X_Cutoff_']), quote=FALSE)
print(paste("confusion_matrix (tp, fp, tn, fn):", stats['X_TP_'], stats['X_FP_'], stats['X_TN_'], stats['X_FN_']), quote=FALSE)