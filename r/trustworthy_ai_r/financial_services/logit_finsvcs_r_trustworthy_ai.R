#################################################################################################
###  Train. Register, Explain, and Assess & Mitigate Bias of R GLM Logistic Regression Model  ###
#################################################################################################

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

conn <- swat::CAS(hostname=hostname, port=port, username, password, protocol=protocol)
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

########################
### Create Dataframe ###
########################

dm_inputdf <- to.casDataFrame(defCasTable(conn, in_mem_tbl, caslib=caslib))
sapply(dm_inputdf, class)

########################
### Model Parameters ###
########################

### import packages
library(dplyr)

### params
link <- 'logit'

### model manager information
model_name <- 'logit_r_finsvcs'
train_code_name <- 'logit_r_finsvcs.r'
score_code_name <- 'logit_r_finsvcs_score.r'
project_name <- 'Financial Services'
description <- 'logit_r'
model_type <- 'Logistic Regression'
metadata_output_dir <- 'outputs'

### define macro variables for model
dm_dec_target <- 'event_indicator'
dm_partitionvar <- 'analytic_partition'
create_new_partition <- 'no' # 'yes'/'no'
dm_key <- 'account_id' 
dm_classtarget_level <- list('0', '1')
dm_partition_val <- list('dm_partition_validate_val'=0, 'dm_partition_train_val'=1, 'dm_partition_test_val'=2)
dm_partition_perc <- list('dm_partition_validate_perc'=0.3, 'dm_partition_train_perc'=0.6, 'dm_partition_test_perc'=0.1)
avg_prob <- mean(dm_inputdf[[dm_dec_target]], na.rm = TRUE)

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


### vars to consider in partial dependency
pd_var <- dm_input

### vars to consider in bias assessment
bias_var <- list('gender', 'at_current_job_1_year', 'job_in_education', 'job_in_hospitality', 'num_dependents')

### vars to consider in bias mitigation
mitigate_var <- list('gender')


### create train, test, validate datasets using existing partition column
set.seed(12345)
split_data <- sample(c(rep(dm_partition_val[1], dm_partition_perc[[1]] * nrow(dm_inputdf)), 
                       rep(dm_partition_val[2], dm_partition_perc[[2]] * nrow(dm_inputdf)), 
                       rep(dm_partition_val[3], dm_partition_perc[[3]] * nrow(dm_inputdf))))
dm_traindf <- dm_inputdf[split_data == dm_partition_val[[2]], ]
train <- subset(dm_traindf, select=c(dm_input, dm_dec_target))
X_train <- subset(dm_traindf, select=dm_input)
y_train <- subset(dm_traindf, select=dm_dec_target)
dm_testdf <- dm_inputdf[split_data == dm_partition_val[[3]], ]
test <- subset(dm_testdf, select=c(dm_input, dm_dec_target))
X_test <- subset(dm_testdf, select=dm_input)
y_test <- subset(dm_testdf, select=dm_dec_target)
dm_validdf <- dm_inputdf[split_data == dm_partition_val[[1]], ]
valid <- subset(dm_validdf, select=c(dm_input, dm_dec_target))
X_valid <- subset(dm_validdf, select=dm_input)
y_valid <- subset(dm_validdf, select=dm_dec_target)

#####################
### Training Code ###
#####################

### estimate & fit model
dm_model <- glm(as.formula(paste(dm_dec_target, " ~ .")), data=train, family=binomial(link=link))

### score full data
full <- subset(dm_inputdf, select=c(dm_dec_target, dm_input))
fullX <- subset(dm_inputdf, select=dm_input)
fully <- subset(dm_inputdf, select=dm_dec_target)
dm_scoreddf_prob_event <- data.frame(predict(dm_model, newdata = full, type = 'response'))
dm_scoreddf_prob_nonevent <- data.frame(1-predict(dm_model, newdata = full, type = 'response'))
dm_scoreddf_class <- data.frame(ifelse(dm_scoreddf_prob_event[[1]] >= avg_prob, 1, 0))
dm_scoreddf <- cbind(dm_scoreddf_prob_nonevent, dm_scoreddf_prob_event, dm_scoreddf_class)
names(dm_scoreddf) <- c(dm_predictionvar[[1]], dm_predictionvar[[2]], dm_classtarget_intovar[[1]])

### create tables with predicted values
trainProba <- data.frame(predict(dm_model, newdata = X_train, type = 'response'))
testProba <- data.frame(predict(dm_model, newdata = X_test, type = 'response'))
validProba <- data.frame(predict(dm_model, newdata = X_valid, type = 'response'))
trainData <- cbind(y_train, dm_classtarget_intovar=trainProba)
testData <- cbind(y_test, dm_classtarget_intovar=testProba)
validData <- cbind(y_valid, dm_classtarget_intovar=validProba)
names(trainData) <- c(dm_dec_target, dm_predictionvar[[2]])
names(testData) <- c(dm_dec_target, dm_predictionvar[[2]])
names(validData) <- c(dm_dec_target, dm_predictionvar[[2]])

### print model & results
summary(dm_model)

#################################
### Register to Model Manager ###
#################################

library(jsonlite)
library(sasctl)
library(pmml)
library(XML)
library(zip)

### define macro vars for model manager metadata script
inputData <- dm_inputdf
trainData <- train
testData <- test
targetVar <- dm_dec_target
intervalVars <- dm_input
analysisPrefix <- description
threshPredProb <- avg_prob
typeOfColumn <- as.data.frame(do.call(rbind, lapply(inputData, typeof)))
fitted.prob <- predict(dm_model, newdata = X_train, type = 'response')
trainData[[targetVar]] <- as.factor(trainData[[targetVar]])

### create directories for metadata
output_path <- file.path(output_dir, metadata_output_dir, model_name)
if (file.exists(output_path)) {
  unlink(output_path, recursive=TRUE) }

### create output path
dir.create(output_path)
analysisFolder <- paste(output_path, '/', sep = '')
jsonFolder <- paste(output_path, '/', sep = '')
zip_folder <- paste(output_path, '/', sep = '')

### create pmml (predictive model markdown language)
pmml_file <- saveXML(pmml(dm_model, model.name = model_name, description = model_type), 
                     paste0(zip_folder, '/', description, '.pmml'))

### move train code and score code to zip directory
file.copy(file.path(output_dir, train_code_name), file.path(output_path, train_code_name))
file.copy(file.path(output_dir, score_code_name), file.path(output_path, score_code_name))

sess <- session(hostname_model, username=username, password=password)

rm <- register_model(
  session = sess,
  file = paste0(zip_folder, '/', description, '.pmml'),
  name = model_name,
  type = "pmml",
  project = project_name,
  force = FALSE
)

#######################################
### Create Stored DATA Step Program ###
#######################################

### copy data step score code from Model Manager (score.sas)
### may need to replace name literals (" "n) with SAS-friendly column name
### alternatively, if the DATA step itself contains embedded strings,
###    then use alternating single and double quotation marks to designate the inner and outer strings
### may need to alter predicted variable to comply with SAS Viya naming convention (e.g., P_(target)1)
### using single thread limits table to 1 obs; multi-thread may create multiple rows of code

loadActionSet(conn, 'dataStep')

cas.dataStep.runCode(conn,
                     code="
                          data ds_score_code;
                          length DataStepSrc varchar(*);
                          DataStepSrc=
                          '
                          PSCR_WARN = 0;
                          
                          if missing(gender) then do;
                              PSCR_WARN = 1;
                          end;

                          if missing(net_worth) then do;
                              PSCR_WARN = 1;
                          end;
                          
                          if missing(job_in_education) then do;
                              PSCR_WARN = 1;
                          end;
                          if missing(job_in_hospitality) then do;
                              PSCR_WARN = 1;
                          end;
                          if missing(at_current_job_1_year) then do;
                              PSCR_WARN = 1;
                          end;
                          if missing(num_dependents) then do;
                              PSCR_WARN = 1;
                          end;
                          if missing(age) then do;
                              PSCR_WARN = 1;
                          end;
                          if missing(debt_to_income) then do;
                              PSCR_WARN = 1;
                          end;
                          if missing(num_transactions) then do;
                              PSCR_WARN = 1;
                          end;
                          if missing(credit_history_mos) then do;
                              PSCR_WARN = 1;
                          end;
                          if missing(credit_score) then do;
                              PSCR_WARN = 1;
                          end;
                          if missing(amount) then do;
                              PSCR_WARN = 1;
                          end;

                          if (PSCR_WARN) then do;
                              goto PSCR_EXIT ;
                          end ;
                          
                          P_event_indicator1 = 0 ;
                          
                          array PSCR_VECTOR_X [ 13 ] PSCR_VECTOR_X0 - PSCR_VECTOR_X12;
                          
                          PSCR_VECTOR_X0 = 1 ;
                          
                          PSCR_VECTOR_X1 = 1 ;
                          
                          PSCR_VECTOR_X2 = 1 ;
                          
                          PSCR_VECTOR_X3 = 1 ;
                          
                          PSCR_VECTOR_X4 = 1 ;
                          
                          PSCR_VECTOR_X5 = 1 ;
                          
                          PSCR_VECTOR_X6 = 1 ;
                          
                          PSCR_VECTOR_X7 = 1 ;
                          
                          PSCR_VECTOR_X8 = 1 ;
                          
                          PSCR_VECTOR_X9 = 1 ;
                          
                          PSCR_VECTOR_X10 = 1 ;
                          
                          PSCR_VECTOR_X11 = 1 ;
                          
                          PSCR_VECTOR_X12 = 1 ;
                          
                          if(not missing (gender)) then do ;
                              PSCR_VECTOR_X1 = PSCR_VECTOR_X1 * gender ;
                          end ;
                          if(not missing (net_worth)) then do ;
                              PSCR_VECTOR_X2 = PSCR_VECTOR_X2 * net_worth ;
                          end ;
                          if(not missing (job_in_education)) then do ;
                              PSCR_VECTOR_X3 = PSCR_VECTOR_X3 * job_in_education ;
                          end ;
                          if(not missing (job_in_hospitality)) then do ;
                              PSCR_VECTOR_X4 = PSCR_VECTOR_X4 * job_in_hospitality ;
                          end ;
                          if(not missing (at_current_job_1_year)) then do ;
                              PSCR_VECTOR_X5 = PSCR_VECTOR_X5 * at_current_job_1_year ;
                          end ;
                          if(not missing (num_dependents)) then do ;
                              PSCR_VECTOR_X6 = PSCR_VECTOR_X6 * num_dependents ;
                          end ;
                          if(not missing (age)) then do ;
                              PSCR_VECTOR_X7 = PSCR_VECTOR_X7 * age ;
                          end ;
                          if(not missing (debt_to_income)) then do ;
                              PSCR_VECTOR_X8 = PSCR_VECTOR_X8 * debt_to_income ;
                          end ;
                          if(not missing (num_transactions)) then do ;
                              PSCR_VECTOR_X9 = PSCR_VECTOR_X9 * num_transactions ;
                          end ;
                          if(not missing (credit_history_mos)) then do ;
                              PSCR_VECTOR_X10 = PSCR_VECTOR_X10 * credit_history_mos ;
                          end ;
                          if(not missing (credit_score)) then do ;
                              PSCR_VECTOR_X11 = PSCR_VECTOR_X11 * credit_score ;
                          end ;
                          if(not missing (amount)) then do ;
                              PSCR_VECTOR_X12 = PSCR_VECTOR_X12 * amount ;
                          end ;
                          PSCR_RESP_CAT = PSCR_VECTOR_X0 * -0.54850157273426 + PSCR_VECTOR_X1 * -1.55181182316368 + PSCR_VECTOR_X2 * -4.45466567327138 + PSCR_VECTOR_X3 * 2.10237899183968 + PSCR_VECTOR_X4 * -1.66540850376675 + PSCR_VECTOR_X5 * -3.21820370807218 +
                           PSCR_VECTOR_X6 * -1.09844709101239 + PSCR_VECTOR_X7 * 0.29562678761402 + PSCR_VECTOR_X8 * 0.59913810075129 + PSCR_VECTOR_X9 * -0.63468822037213 + PSCR_VECTOR_X10 * -0.81818653193204 + PSCR_VECTOR_X11 * -1.25194540643588 + PSCR_VECTOR_X12 *
                           -0.51355618178663 ; 
                          PSCR_OFFSET = 0; 
                          PSCR_TRIAL = 1; 
                          P_event_indicator1 = PSCR_RESP_CAT + PSCR_OFFSET;
                          P_event_indicator1 = 1 / (1 + exp (-(max(min(P_event_indicator1,500),-500)) )) ;
                          
                          P_event_indicator1 = P_event_indicator1 * PSCR_TRIAL;
                          PSCR_EXIT :
                          
                          drop
                           PSCR_VECTOR_X0 PSCR_VECTOR_X1 PSCR_VECTOR_X2 PSCR_VECTOR_X3 PSCR_VECTOR_X4 PSCR_VECTOR_X5 PSCR_VECTOR_X6 PSCR_VECTOR_X7 PSCR_VECTOR_X8 PSCR_VECTOR_X9 PSCR_VECTOR_X10 PSCR_VECTOR_X11 PSCR_VECTOR_X12 PSCR_RESP_CAT PSCR_OFFSET PSCR_TRIAL
                           PSCR_WARN;
                          '
                          ; 
                          run;
                          "
                     , single="yes"
                    )

results <- cas.table.fetch(conn, table=list(name="ds_score_code")) 
results

score_code_tbl <- 'ds_score_code'
score_code <- to.casDataFrame(defCasTable(conn, 'ds_score_code'))

######################################################################################
### Trustworthy AI - Partial Dependence, SHAP Values, Assess Bias, Bias Mitigation ###
######################################################################################

loadActionSet(conn, 'explainModel')
loadActionSet(conn, 'fairAITools')

##########################
### Partial Dependence ###
##########################

library('stringr')

part_depend_size <- length(pd_var)
part_depend_tbl <- paste(model_name, "_", "partial_dependence", sep="")

if (cas.table.tableExists(conn, name=part_depend_tbl)>0) {
  cas.table.dropTable(conn, name=part_depend_tbl)
  cas.table.dropTable(conn, name=part_depend_tbl, quiet=TRUE)
}

for (i in 1:part_depend_size) {
  cas.explainModel.partialDependence(conn,
                                     table=list(caslib=caslib, name=in_mem_tbl),
                                     seed=12345,
                                     modelTable=list(name=score_code_tbl),
                                     modelTableType="DATASTEP",
                                     predictedTarget=dm_predictionvar[[2]],
                                     analysisVariable=list(name=pd_var[[i]], nBins=20),
                                     inputs=dm_input,
                                     outputTables=list(names=list(PartialDependence=list(name='partialdependence', 
                                                                                         replace=TRUE)))
  )
  
  string_format <- str_pad(string=pd_var[[i]], width=32, side='right', pad=' ')
  var_name <- paste("'", string_format, "'", sep="")
  program <- paste("variable=",var_name) 
  # computedVarsProgram has a very specific format, and format length needs to be exactly the same in each table to append #
  cas.table.copyTable(conn,
                      table=list(name='partialdependence',
                                 computedVars=list(list(name='variable', FormattedLength=32)),
                                 computedVarsProgram=program),
                      casOut=list(name='partialdependence', replace=TRUE)
  )
  
  cas.table.alterTable(conn,
                       name='partialdependence',
                       columns=list(list(name=pd_var[[i]], rename='bin_value', label='bin_value'),
                                    list(name='MeanPrediction', rename='mean_prediction', label='mean_prediction'),
                                    list(name='StdErr', rename='standard_error', label='standard_error'),
                                    list(name='Bin', rename='bin_num', label='bin_num')
                       )
  )
  
  if (i == 1) {
    cas.table.copyTable(conn,
                        table=list(name='partialdependence'),
                        casOut=list(name=part_depend_tbl, replace=TRUE)
    )
  }
  else {
    cas.table.append(conn,
                     source=list(name='partialdependence'),
                     target=list(name=part_depend_tbl)
    )
  }
}

partial_dependence <- to.casDataFrame(defCasTable(conn, table=part_depend_tbl))
cas.table.columnInfo(conn, table=list(name=part_depend_tbl))

cas.table.promote(conn, caslib='casuser', name=part_depend_tbl, drop=TRUE)

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
                 output=list(casOut=list(name='sample_out', replace=TRUE), copyVars=list(dm_key)),
)
rand_obs <- array(to.casDataFrame(defCasTable(conn, table='sample_out')))
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
                                    modelTable=list(name=score_code_tbl),
                                    modelTableType="DATASTEP",
                                    predictedTarget=dm_predictionvar[[2]],
                                    inputs=dm_input,
                                    depth=1,
                                    outputTables=list(names=list(ShapleyValues=list(name='shapleyvalues', 
                                                                                    caslib=caslib, replace=TRUE)))
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

### note that the input table is a scored table ###
### assess bias action produces local output ###

dm_scoreddf_bias <- subset(dm_inputdf, select=c(dm_dec_target, unlist(bias_var)))
scored <- cbind(dm_scoreddf, dm_scoreddf_bias)
scored_data <- as.casTable(conn, scored, casOut=list(name='scored_tbl'))

rm('group_metrics', 'group_metrics_all')
rm('max_differences', 'max_differences_all')
rm('bias_metrics', 'bias_metrics_all')

bias_size <- length(bias_var)
group_metrics_tbl <- paste(model_name, "_", "group_metrics", sep="")
max_differences_tbl <- paste(model_name, "_", "max_differences", sep="")
bias_metrics_tbl <- paste(model_name, "_", "bias_metrics", sep="")

for (i in 1:bias_size) {
  assess_bias_results <- 
    cas.fairAITools.assessBias(conn,
                               table = 'scored_tbl',
                               modelTableType = "NONE",
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

### promotes R dataframe to global scope CAS table ###
as.casTable(conn, group_metrics_all, casOut=list(caslib='casuser', name=group_metrics_tbl, promote=TRUE))
as.casTable(conn, max_differences_all, casOut=list(caslib='casuser', name=max_differences_tbl, promote=TRUE))
as.casTable(conn, bias_metrics_all, casOut=list(caslib='casuser', name=bias_metrics_tbl, promote=TRUE))

#####################
### Mitigate Bias ###
#####################

### mitigateBias action is used to run the EGR algorithm to train a model with reduced bias (biasMetric)
###   calculated on the sensitive variable(s) in the mitigate_var list

### model arguments
### the 'trainProgram' parameter is CASL, not R or Python API

event <- 'LAST'
selection_method <- 'STEPWISE'
cas_out_tbl <- "mitigate_bias_model"
astore_tbl <- "mitigate_bias_astore"
cas_score_tbl <- "mitigate_bias_score"

mitigate_size <- length(mitigate_var)

for (i in 1:mitigate_size) {
  mitigate_bias_results <-
    cas.fairAITools.mitigateBias(conn,
                             table = list(caslib=caslib, name=in_mem_tbl),
                             response = dm_dec_target,
                             predictedVariables = list(dm_predictionvar[[1]], dm_predictionvar[[2]]),
                             responseLevels = dm_classtarget_level,
                             sensitiveVariable = mitigate_var[[i]],
                             biasMetric = 'DEMOGRAPHICPARITY',
                             event = '1',
                             learningRate = '0.01',
                             maxIters = '10',
                             tolerance = '0.005',
                             tuneBound = 'True',
                             trainProgram = '
                                regression.logistic result=mitigate_model_train /
                                  table = {caslib=caslib, name=in_mem_tbl},
                                  model = {depVars={name=dm_dec_target, options={event=event}}, effects={vars=dm_input,informative=True}},
                                  partByVar = {name=dm_partitionvar, train=str(dm_partition_train_val), valid=str(dm_partition_validate_val), test=str(dm_partition_test_val)},
                                  output = {casOut={caslib="casuser", name=cas_out_tbl, replace=True}, copyVars=dm_dec_target, into=dm_classtarget_intovar},
                                  selection = {method=selection_method},
                                  savestate = {caslib="casuser", name=astore_tbl, replace=True}
                                  ;
                                astore.score result=mitigate_model_score /
                                  table = {caslib="casuser", name=in_mem_tbl},
                                  casOut={caslib="casuser", name=cas_score_tbl}
                                  copyVars={dm_dec_target, dm_partitionvar},
                                  rstore={caslib="casuser", name=astore_tbl}
                                  ;
                            '
                              )
                          }


mitigate_bias_results <-
  cas.fairAITools.mitigateBias(conn,
                               table = 'financial_services_prep',
                               response = 'event_indicator',
                               predictedVariables = c('P_event_indicator0', 'P_event_indicator1'),
                               responseLevels = c('0', '1'),
                               sensitiveVariable = 'gender',
                               biasMetric = 'DEMOGRAPHICPARITY',
                               event = '1',
                               learningRate = '0.01',
                               maxIters = '10',
                               tolerance = '0.005',
                               tuneBound = TRUE,
                               trainProgram = '
                                regression.logistic result=mitigate_model_train /
                                  table = table,
                                  weight = weight,
                                  model = {depVars={name="event_indicator", options={event="LAST"}}, 
                                           effects={vars= {"at_current_job_1_year", "num_dependents",
                                                          "age", "amount", "credit_history_mos", "credit_score",
                                                          "debt_to_income", "net_worth", "num_transactions"}, 
                                                          informative=True
                                                    },
                                  savestate = "mitigate_bias_astore"
                                  ;
                                astore.score result=mitigate_model_score /
                                  table = table,
                                  casOut = casout,
                                  copyVars = copyVars,
                                  rstore = "mitigate_bias_astore"
                                  ;
                            '
  )


results <- cas.table.fetch(conn, table=list(caslib="casuser", name="mitigate_model_train")) 
results
