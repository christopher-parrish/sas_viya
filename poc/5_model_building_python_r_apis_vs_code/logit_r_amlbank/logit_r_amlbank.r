
###################
### Credentials ###
###################

library(sys)

wd <- "C:/Users/chparr/OneDrive - SAS/credentials"
setwd(wd)
source(file.path(wd, 'credentials.r'))

###################
### Environment ###
###################

library(swat)

access_token <- token
conn <- swat::CAS(hostname=hostname, port=port, username=NULL, password=access_token, protocol=protocol)
print(conn)
print(cas.builtins.serverStatus(conn))

#############################
### Identify Table in CAS ###
#############################

### caslib and table to use in scoring
caslib <- 'casuser'
in_mem_tbl <- 'AML_BANK_PREP'

### load table in-memory if not already exists in-memory ###
if (cas.table.tableExists(conn, caslib=caslib, name=in_mem_tbl)<=0) {
  cas.table.loadTable(conn, caslib=caslib, path=paste0(in_mem_tbl,('.sashdat')), casout=list(name=in_mem_tbl, caslib=caslib, promote=TRUE))}

### show table to verify
cas.table.tableInfo(conn, caslib=caslib, wildIgnore=FALSE, name=in_mem_tbl)

########################
### Create Dataframe ###
########################

dm_inputdf <- to.casDataFrame(defCasTable(conn, in_mem_tbl, caslib=caslib))
sapply(dm_inputdf, class)

### for csv; may have to address bad characters in first column
# dm_inputdf <- read.table(file.path(data_dir, 'aml_bank_prep.csv'), header = TRUE, sep = ',')
# colnames(dm_inputdf)[1] <- gsub('^...','',colnames(dm_inputdf)[1])

########################
### Model Parameters ###
########################

### import packages
library(dplyr)

### params
link <- 'logit'

### model manager information
modeler <- 'Parrish'
metadata_output_dir <- 'outputs'
model_name <- 'logit_r_amlbank'
train_code_name <- 'logit_r_amlbank.r'
score_code_name <- 'logit_r_amlbank_score.r'
project_name <- 'Anti-Money Laundering'
description <- 'Logistic Regression'
model_type <- 'Logistic Regression'

### define macro variables for model
dm_dec_target <- 'ml_indicator'
dm_partitionvar <- 'analytic_partition' 
dm_key <- 'account_id' 
dm_classtarget_level <- list('0', '1')
dm_partition_validate_val <- 0
dm_partition_validate_perc <- 0.3
dm_partition_train_val <- 1
dm_partition_train_perc <- 0.6
dm_partition_test_val <- 2
dm_partition_test_perc <- 0.1
avg_prob <- mean(dm_inputdf[[dm_dec_target]], na.rm = TRUE)

### create list of rejected predictor columns
rejected_predictors <- list(
  'atm_deposit_indicator', 
  'citizenship_country_risk', 
  'distance_to_bank',
  'distance_to_employer', 
  'income', 
  'num_acctbal_chgs_gt2000',
  'occupation_risk',
  'marital_status_single',
  'primary_transfer_cash')

##############################
### Final Modeling Columns ###
##############################

### create list of model variables
dm_input <- colnames(dm_inputdf)
macro_vars <- list(dm_dec_target, dm_partitionvar, dm_key)
rejected_vars <- unlist(c(rejected_predictors, macro_vars))
dm_input <- dm_input[! dm_input %in% c(rejected_vars)]
dm_input
  
### create prediction variables
dm_predictionvar <- list((paste("P_", dm_dec_target, dm_classtarget_level[1], sep ="")),
                         (paste("P_", dm_dec_target, dm_classtarget_level[2], sep ="")))
dm_classtarget_intovar <- (paste("I_", dm_dec_target, sep =""))

##################
### Data Split ###
##################

### create train, test, validate datasets using existing partition column
set.seed(12345)
split_data <- sample(c(rep(dm_partition_validate_val, dm_partition_validate_perc * nrow(dm_inputdf)), 
                       rep(dm_partition_train_val, dm_partition_train_perc * nrow(dm_inputdf)), 
                       rep(dm_partition_test_val, dm_partition_test_perc * nrow(dm_inputdf))))
dm_traindf <- dm_inputdf[split_data == dm_partition_train_val, ]
train <- subset(dm_traindf, select=c(dm_input, dm_dec_target))
X_train <- subset(dm_traindf, select=dm_input)
y_train <- subset(dm_traindf, select=dm_dec_target)
dm_testdf <- dm_inputdf[split_data == dm_partition_test_val, ]
test <- subset(dm_testdf, select=c(dm_input, dm_dec_target))
X_test <- subset(dm_testdf, select=dm_input)
y_test <- subset(dm_testdf, select=dm_dec_target)
dm_validdf <- dm_inputdf[split_data == dm_partition_validate_val, ]
valid <- subset(dm_validdf, select=c(dm_input, dm_dec_target))
X_valid <- subset(dm_validdf, select=dm_input)
y_valid <- subset(dm_validdf, select=dm_dec_target)

#####################
### Training Code ###
#####################

### estimate & fit model
dm_model <- glm(ml_indicator ~ ., train, family=binomial(link=link))

### score full data
full <- subset(dm_inputdf, select=c(dm_dec_target, dm_input))
fullX <- subset(dm_inputdf, select=dm_input)
fully <- subset(dm_inputdf, select=dm_dec_target)
dm_scoreddf_prob_event <- data.frame(predict(dm_model, newdata = full, type = 'response'))
dm_scoreddf_prob_nonevent <- data.frame(1-predict(dm_model, newdata = full, type = 'response'))
dm_scoreddf_class <- data.frame(ifelse(dm_scoreddf_prob_event[1] >= avg_prob, 1, 0))
dm_scoreddf <- cbind(dm_scoreddf_prob_nonevent, dm_scoreddf_prob_event, dm_scoreddf_class)
names(dm_scoreddf) <- c(dm_predictionvar[1], dm_predictionvar[2], dm_classtarget_intovar)

### create tables with predicted values
trainProba <- data.frame(predict(dm_model, newdata = X_train, type = 'response'))
testProba <- data.frame(predict(dm_model, newdata = X_test, type = 'response'))
validProba <- data.frame(predict(dm_model, newdata = X_valid, type = 'response'))
trainData <- cbind(y_train, dm_classtarget_intovar=trainProba)
testData <- cbind(y_test, dm_classtarget_intovar=testProba)
validData <- cbind(y_valid, dm_classtarget_intovar=validProba)
names(trainData) <- c(dm_dec_target, dm_predictionvar[2])
names(testData) <- c(dm_dec_target, dm_predictionvar[2])
names(validData) <- c(dm_dec_target, dm_predictionvar[2])

### print model & results
summary(dm_model)

#######################################
### Register Model in Model Manager ###
## Ensure Model Does Not Exist in MM ##
#######################################

library(jsonlite)
library(sasctl)
library(zip)

### create directories for metadata
output_path <- file.path(output_dir, metadata_output_dir, model_name)
if (file.exists(output_path)) {
  unlink(output_path, recursive=TRUE) }

### create output path
dir.create(output_path)
output_path_folder <- paste(output_path, '/', sep = '')

### create metadata
saveRDS(dm_model, paste0(output_path_folder, paste0(model_name, '.rds')))
score_code <- codegen(dm_model, path = paste0(output_path_folder, score_code_name), rds = paste0(model_name, '.rds'))
#file.show(paste0(output_path_folder,"/scoreCode.R"), title=NULL)
#create_scoreSample(output_path_folder, openFile = FALSE)

write_ModelProperties_json(modelName = model_name, 
                           modelDescription = description, 
                           modelFunction = "Classification",
                           trainTable = in_mem_tbl,
                           algorithm = model_type,
                           numTargetCategories = 2,
                           targetEvent = "1",
                           targetVariable = dm_dec_target,
                           eventProbVar = dm_predictionvar[[2]],
                           modeler = username,
                           tool = "R",
                           toolVersion = "default",
                           path = output_path_folder)

write_in_out_json(data=dm_traindf[,], 
                  input=TRUE, 
                  path=output_path_folder)

write_in_out_json(data=dm_scoreddf[,], 
                  input=FALSE, 
                  path=output_path_folder)

write_fileMetadata_json(scoreCodeName = score_code_name,
                        scoreResource = paste0(model_name, '.rds'),
                        path = output_path_folder)

diag = diagnosticsJson(validadedf = validData,
                       traindf = trainData,
                       testdf = testData,
                       targetEventValue = 1,
                       targetName = dm_dec_target,
                       targetPredicted = dm_predictionvar[[2]],
                       path = output_path_folder)

### copy train script to output path
file.copy(file.path(paste0(git_dir, '/r/logit_r/aml_bank'), train_code_name), file.path(output_path_folder, train_code_name))

zip_files = list.files(output_path_folder, full.names = T)
zipr(zipfile=paste0(output_path_folder, paste0(model_name, '.zip')), files=zip_files)

### create session in cas
sess <- session(hostname=session, oauth_token=token)

rmodel <- register_model(
  session = sess,
  file = paste0(output_path_folder, paste0(model_name, '.zip')),
  name = model_name,
  type = "zip",
  project = project_name,
  force = FALSE
)

#######################
### Create Metadata ###
#######################

library(jsonlite)

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

### move train code and score code to zip directory
file.copy(file.path(output_dir, train_code_name), file.path(output_path, train_code_name))
file.copy(file.path(output_dir, score_code_name), file.path(output_path, score_code_name))

### create metadata for import to model manager
source(file.path(wd, 'export_binary_model.r'))
export_binary_model (
  targetVar = targetVar,
  intervalVars = intervalVars,
  nominalVars = NULL,
  nominalVarsLength = NULL,
  typeOfColumn = typeOfColumn,
  targetValue = trainData[[targetVar]],
  eventValue = 1,
  predEventProb = fitted.prob[],
  eventProbThreshold = threshPredProb,
  algorithmCode = 1,
  modelObject = dm_model,
  analysisFolder = analysisFolder,
  analysisPrefix = analysisPrefix,
  jsonFolder = jsonFolder,
  analysisName = project_name,
  analysisDescription = model_type,
  lenOutClass = 1,
  qDebug = 'Y')

# algorithmCode(1,2,3,4,'') = ('Logistic regression', 'Discriminant', 'Decision tree', 'Gradient boosting', NA)
