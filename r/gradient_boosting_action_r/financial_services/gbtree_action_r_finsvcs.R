
######################################################
###  Train & Register SAS Gradient Boosting Model  ###
######################################################

###################
### Credentials ###
###################

library(askpass)
library(sys)

username <- askpass("USERNAME")
password <- askpass("PASSWORD")
wd <- askpass("What is the Working Directory for this R Session?")
source(file.path(wd, 'password.r'))
metadata_output_dir <- 'outputs'

###################
### Environment ###
###################

library(swat)

#conn <- swat::CAS(hostname=hostname, port=port, username, password, protocol=protocol)
conn <- CAS(hostname_sse, password=token_sse, protocol=protocol_sse) # password=token_sse
print(cas.builtins.serverStatus(conn))

#############################
### Identify Table in CAS ###
#############################

### caslib and table to use in modeling
caslib <- 'Public'
in_mem_tbl <- 'FINANCIAL_SERVICES_PREP'

### load table in-memory if not already exists in-memory
if (cas.table.tableExists(conn, caslib=caslib, name=in_mem_tbl)<=0) {
  cas.table.loadTable(conn, caslib=caslib, path=paste(in_mem_tbl,('.sashdat'), sep = ""), 
                      casout=list(name=in_mem_tbl, caslib=caslib, promote=TRUE))}

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
model_name <- 'gbtree_action_r_finsvcs'
project_name <- 'Financial Services'
description <- 'gbtree_r'
model_type <- 'gradient_boosting'

### define macro variables for model
dm_dec_target <- 'event_indicator'
dm_partitionvar <- 'analytic_partition'
create_new_partition <- 'no' # 'yes'/'no'
dm_key <- 'account_id' 
dm_classtarget_level <- list('0', '1')
dm_partition_val <- list('dm_partition_validate_val'=0, 'dm_partition_train_val'=1, 'dm_partition_test_val'=2)
dm_partition_perc <- list('dm_partition_validate_perc'=0.3, 'dm_partition_train_perc'=0.6, 'dm_partition_test_perc'=0.1)

##############################
### Final Modeling Columns ###
##############################

### create list of model variables
dm_input <- colnames(dm_inputdf)
macro_vars <- list(dm_dec_target, dm_partitionvar, dm_key)
keep_predictors <- list(
  'net_worth',
  'credit_score',
  'num_dependents',
  'at_current_job_1_year',
  'credit_history_mos',
  'job_in_education',
  'num_transactions',
  'debt_to_income',
  'amount',
  'gender',
  'age',
  'job_in_hospitality'
)
#keep_predictors <- 
#rejected_predictors <- list()
rejected_predictors <- dm_input[! dm_input %in% c(keep_predictors)]
rejected_vars <- unlist(c(rejected_predictors)) # , macro_vars
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

### vars to consider in bias assessment
bias_var <- list('gender', 'age')

### vars to consider in partial dependency
pd_var <- dm_input #list('credit_score', 'net_worth')

#####################
### Training Code ###
#####################

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

####################
###  Score Data  ###
####################

cas.decisionTree.gbtreeScore(conn,
  modelTable=list(caslib=caslib, name=cas_out_tbl),
  table=list(caslib=caslib, name=in_mem_tbl), 
  copyvars=list(dm_dec_target, dm_partitionvar),
  casout=list(caslib=caslib, name=cas_score_tbl, replace=TRUE),
  encodeName=TRUE,
  assessOneRow=TRUE
  )

####################
### Scoring Code ###
####################

cas.decisionTree.gbtreeCode(conn,
  modelTable=list(caslib=caslib, name=cas_out_tbl),
  code=list(casOut=list(caslib=caslib, name='gbtree_scorecode', replace=TRUE, promote=FALSE))
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

#####################################################################
### Trustworthy AI - Partial Dependence, SHAP Values, Assess Bias ###
#####################################################################

##########################
### Partial Dependence ###
##########################

library('stringr')

part_depend_size <- length(pd_var)
part_depend_tbl <- paste(model_name, "_", "partial_dependence", sep="")

if (cas.table.tableExists(conn, caslib=caslib, name=part_depend_tbl)>0) {
  cas.table.dropTable(conn, caslib=caslib, name=part_depend_tbl)
  cas.table.dropTable(conn, caslib=caslib, name=part_depend_tbl, quiet=TRUE)
}

for (i in 1:part_depend_size) {
cas.explainModel.partialDependence(conn,
    table=list(caslib=caslib, name=in_mem_tbl),
    seed=12345,
    modelTable=list(caslib=caslib, name=astore_tbl),
    predictedTarget=dm_predictionvar[[2]],
    analysisVariable=list(name=pd_var[[i]], nBins=20),
    inputs=dm_input,
    outputTables=list(names=list(PartialDependence=list(name='partialdependence', caslib=caslib, replace=TRUE)))
    )

  string_format <- str_pad(string=pd_var[[i]], width=32, side='right', pad=' ')
  var_name <- paste("'", string_format, "'", sep="")
  program <- paste("variable=",var_name) 
  # computedVarsProgram has a very specific format, and format length needs to be exactly the same in each table to append #
  cas.table.copyTable(conn,
                      table=list(caslib=caslib, name='partialdependence',
                                 computedVars=list(list(name='variable', FormattedLength=32)),
                                 computedVarsProgram=program),
                      casOut=list(caslib=caslib, name='partialdependence', replace=TRUE)
  )

  cas.table.alterTable(conn,
                       caslib=caslib, name='partialdependence',
                       columns=list(list(name=pd_var[[i]], rename='bin_value', label='bin_value'),
                                    list(name='MeanPrediction', rename='mean_prediction', label='mean_prediction'),
                                    list(name='StdErr', rename='standard_error', label='standard_error'),
                                    list(name='Bin', rename='bin_num', label='bin_num')
                                    )
  )

  if (i == 1) {
    cas.table.copyTable(conn,
                        table=list(caslib=caslib, name='partialdependence'),
                        casOut=list(caslib=caslib, name=part_depend_tbl, replace=TRUE)
    )
  }
  else {
    cas.table.append(conn,
                     source=list(caslib=caslib, name='partialdependence'),
                     target=list(caslib=caslib, name=part_depend_tbl)
    )
  }
}

partial_dependence <- to.casDataFrame(defCasTable(conn, caslib=caslib, table=part_depend_tbl))
cas.table.columnInfo(conn, table=list(caslib=caslib, name=part_depend_tbl))

cas.table.promote(conn, caslib=caslib, name=part_depend_tbl, drop=TRUE)

###################
### SHAP Values ###
###################

loadActionSet(conn, 'transpose')
loadActionSet(conn, 'sampling')

sample_size <- 50
cas.sampling.srs(conn,
                 fixedObs=sample_size,
                 seed=12345,
                 table=list(caslib=caslib, name=in_mem_tbl),
                 output=list(casOut=list(caslib=caslib, name='sample_out', replace=TRUE), copyVars=list(dm_key)),
                 )
rand_obs <- array(to.casDataFrame(defCasTable(conn, caslib=caslib, table='sample_out')))
shapley_rows_tbl <- paste(model_name, "_", "shapley_rows", sep="")
shapley_cols_tbl <- paste(model_name, "_", "shapley_cols", sep="")

### drop session scoped AND global scoped tables if they exist ###
if (cas.table.tableExists(conn, caslib=caslib, name=shapley_rows_tbl)>0) {
  cas.table.dropTable(conn, caslib=caslib, name=shapley_rows_tbl)
  cas.table.dropTable(conn, caslib=caslib, name=shapley_rows_tbl, quiet=TRUE)
}
if (cas.table.tableExists(conn, caslib=caslib, name=shapley_cols_tbl)>0) {
  cas.table.dropTable(conn, caslib=caslib, name=shapley_cols_tbl)
  cas.table.dropTable(conn, caslib=caslib, name=shapley_cols_tbl, quiet=TRUE)
}

for (i in 1:sample_size) {
  obs_num <- as.numeric(rand_obs[i,])
  #shapley_temp <- dm_inputdf[dm_inputdf[dm_key] == obs_num,]
  query_part = paste(dm_key, '=', obs_num, sep="")

  cas.explainModel.shapleyExplainer(conn,
    table=list(caslib=caslib, name=in_mem_tbl),
    query=list(caslib=caslib, name=in_mem_tbl, where=query_part),
    modelTable=list(caslib=caslib, name=astore_tbl),
    modelTableType="astore",
    predictedTarget=dm_predictionvar[[2]],
    inputs=dm_input,
    depth=1,
    outputTables=list(names=list(ShapleyValues=list(name='shapleyvalues', caslib=caslib, replace=TRUE)))
  )

  cas.transpose.transpose(conn,
    table=list(caslib=caslib, name="shapleyvalues"),
    id="variable",
    label=dm_key,
    casOut=list(caslib=caslib, name="shapley_transpose", replace=TRUE)
  )
  
  cas.table.update(conn, 
                   table=list(caslib=caslib, name='shapley_transpose'),
                   set=list(list(var=dm_key, value=as.character(obs_num)))
  )
  
  cas.table.alterTable(conn, 
                   caslib=caslib, name='shapley_transpose',
                   drop=list('_NAME_')
  )
  
  if (i == 1) {
    cas.table.copyTable(conn,
      table=list(caslib=caslib, name="shapley_transpose"),
      casOut=list(caslib=caslib, name=shapley_rows_tbl, replace=TRUE)
    )
    cas.table.copyTable(conn,
      table=list(caslib=caslib, name="shapleyvalues"),
      casOut=list(caslib=caslib, name=shapley_cols_tbl, replace=TRUE))
  }
  else {
    cas.table.append(conn,
      source=list(caslib=caslib, name='shapley_transpose'),
      target=list(caslib=caslib, name=shapley_rows_tbl)
    )
    cas.table.append(conn, 
      source=list(caslib=caslib, name='shapleyvalues'),
      target=list(caslib=caslib, name=shapley_cols_tbl)
    )
  }
}

shapley_rows <- to.casDataFrame(defCasTable(conn, caslib=caslib, table=shapley_rows_tbl))
shapley_cols <- to.casDataFrame(defCasTable(conn, caslib=caslib, table=shapley_cols_tbl))

cas.table.promote(conn, caslib=caslib, name=shapley_rows_tbl, drop=TRUE)
cas.table.promote(conn, caslib=caslib, name=shapley_cols_tbl, drop=TRUE)

###################
### Assess Bias ###
###################

bias_size <- length(bias_var)
group_metrics_tbl <- paste(model_name, "_", "group_metrics", sep="")
max_differences_tbl <- paste(model_name, "_", "max_differences", sep="")
bias_metrics_tbl <- paste(model_name, "_", "bias_metrics", sep="")

for (i in 1:bias_size) {
assess_bias_results <- 
  cas.fairAITools.assessBias(conn,
    table = list(caslib=caslib, name=in_mem_tbl),
    modelTable = list(caslib=caslib, name=astore_tbl),
    modelTableType = "ASTORE",
    response = dm_dec_target,
    predictedVariables = list(dm_predictionvar[[1]], dm_predictionvar[[2]]),
    responseLevels = dm_classtarget_level,
    sensitiveVariable = bias_var[[i]]
  )

  group_metrics <- as.data.frame(assess_bias_results[1])
  group_metrics['bias_var'] <- bias_var[[i]]
  max_differences <- as.data.frame(assess_bias_results[2])
  max_differences['bias_var'] <- bias_var[[i]]
  bias_metrics <- as.data.frame(assess_bias_results[3])
  bias_metrics['bias_var'] <- bias_var[[i]]

  if (i == 1) {
    group_metrics_all <- group_metrics
    max_differences_all <- max_differences
    bias_metrics_all <- bias_metrics
  }
  else {
    group_metrics_all <- rbind(group_metrics_all, group_metrics)
    max_differences_all <- rbind(max_differences_all, max_differences)
    bias_metrics_all <- rbind(bias_metrics_all, bias_metrics)
  }
}

as.casTable(conn, group_metrics_all, casOut=list(caslib=caslib, name=group_metrics_tbl, promote=TRUE))
as.casTable(conn, max_differences_all, casOut=list(caslib=caslib, name=max_differences_tbl, promote=TRUE))
as.casTable(conn, bias_metrics_all, casOut=list(caslib=caslib, name=bias_metrics_tbl, promote=TRUE))

##################################
### Register Model in SAS Viya ###
##################################

library(sasctl)
astore_blob <- cas.astore.download(conn, rstore=list(caslib=caslib, name=astore_tbl))
## saving astore as binary file
astore_path <- "./rf_model.astore"
con <- file(astore_path, "wb")
### file is downloaded as base64 encoded
writeBin(object=astore_blob$blob, con=con, useBytes=T)
close(con)
sess <- session("https://fsbulab.unx.sas.com", username=username, password=password)
output <- register_model(session = sess,
                           file = astore_path,
                           name = model_name,
                           type = "astore",
                           project = project_name
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
