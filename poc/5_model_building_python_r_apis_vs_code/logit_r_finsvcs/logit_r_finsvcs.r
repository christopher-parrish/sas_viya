
########################################
###  Train & Register R Logit Model  ###
########################################

###################
### Credentials ###
###################

library(askpass)
library(sys)

cred <- askpass("What is the credentials path for this R Session?")
source(file.path(cred, 'credentials.r'))

#############################
### Connect with SAS Viya ###
#############################

library(swat)

conn <- CAS(hostname, password=token, protocol=protocol)
print(cas.builtins.serverStatus(conn))

#############################
### Identify Table in CAS ###
#############################

### caslib and table to use in modeling
caslib <- 'casuser'
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

### print columns for review of model parameters
sapply(dm_inputdf, class)

########################
### Model Parameters ###
########################

### import packages
library(dplyr)

### params
link <- 'logit'

### model manager information
metadata_output_dir <- 'outputs'
model_name <- 'logit_r_finsvcs'
train_code_name <- 'logit_r_finsvcs.r'
score_code_name <- 'logit_r_finsvcs_score.r'
project_name <- 'Financial Services'
description <- 'Logistic Regression'
model_type <- 'Logistic Regression'

### define macro variables for model
dm_dec_target <- 'event_indicator'
dm_partitionvar <- 'analytic_partition'
create_new_partition <- 'no' # 'yes'/'no'
dm_key <- 'account_id' 
dm_classtarget_level <- list('0', '1')
dm_partition_val <- list('dm_partition_validate_val'=0, 'dm_partition_train_val'=1, 'dm_partition_test_val'=2)
dm_partition_perc <- list('dm_partition_validate_perc'=0.3, 'dm_partition_train_perc'=0.6, 'dm_partition_test_perc'=0.1)
avg_prob <- mean(dm_inputdf[[dm_dec_target]], na.rm = TRUE)

### create list of regressors
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
#rejected_predictors <- list()

### vars to consider in bias assessment
bias_var <- list('gender', 'age')

### vars to consider in partial dependency
pd_var <- list('credit_score', 'net_worth')

##############################
### Final Modeling Columns ###
##############################

dm_input <- colnames(dm_inputdf)
macro_vars <- list(dm_dec_target, dm_partitionvar, dm_key)
rejected_predictors <- dm_input[! dm_input %in% c(keep_predictors)]
rejected_vars <- unlist(c(rejected_predictors)) # , macro_vars
dm_input <- dm_input[! dm_input %in% c(rejected_vars)]

### create prediction variables
dm_predictionvar <- list((paste("P_", dm_dec_target, dm_classtarget_level[1], sep ="")),
                         (paste("P_", dm_dec_target, dm_classtarget_level[2], sep ="")),
                         'EM_EVENTPROBABILITY')
dm_classtarget_intovar <- list((paste("I_", dm_dec_target, sep ="")), 'EM_CLASSIFICATION')

##################
### Data Split ###
##################

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
file.copy(file.path(output_dir, train_code_name), file.path(output_path_folder, train_code_name))

zip_files = list.files(zip_folder, full.names = T)
zipr(zipfile=paste0(zip_folder,"/R_Logistic_Model.zip"), files=zip_files)

### create session in cas
sess <- session(hostname=session, oauth_token=token)

rmodel <- register_model(
  session = sess,
  file = paste0(zip_folder,"/R_Logistic_Model.zip"),
  name = "R Logistic Model",
  type = "zip",
  project = "MM_OS_Test",
  force = FALSE
)






### find data step score code url

lm <- list_model_contents(
    session = sess,
    model = model_name
)

list_place <- which(sapply(lm$name, function(x) "score.sas" %in% x))
score_code_uri <- lm$fileUri[list_place]
score_code_id <- lm$id[list_place]



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






### jordan's code ###

output_dir = paste0(dirname(getwd()),"/Model_Manager/Metadata")
model_name = "R_LR_Model"
data_name = "HMEQ"
zip_folder = paste0(output_dir, "/", data_name, "_", model_name)

if (file.exists(zip_folder)){
  unlink(zip_folder, recursive=TRUE) 
}

dir.create(zip_folder)

mypmml = saveXML(pmml(lr, model.name = "Logistic Regression",
                      description = "Logistic Regression Model"),
                 paste0(zip_folder, '/lr_model.pmml'))

df = read.csv(paste0(dirname(getwd()),"/password_r.txt"), header=TRUE, stringsAsFactors=FALSE)

sess = session(hostname=paste0("https://", strsplit(df$hostname,"/")[[1]][1]), username=df$username, password=df$password)

mod = register_model(
  session = sess,
  file = paste0(zip_folder, '/lr_model.pmml'),
  name = "R_Logistic_pmml",
  type = "pmml",
  project = "MM_OS_Test",
  force = FALSE
)

output_dir = paste0(dirname(getwd()),"/Model_Manager/Metadata")
model_name = "R_LR_Model"
data_name = "HMEQ"
zip_folder = paste0(output_dir, "/", data_name, "_", model_name)

if (file.exists(zip_folder)){
  unlink(zip_folder, recursive=TRUE) 
}

dir.create(zip_folder)

saveRDS(lr, paste0(zip_folder, '/lr_model.rds'))

create_scoreSample(zip_folder, openFile = FALSE)
file.show(paste0(zip_folder,"/scoreCode.R"), title=NULL)

write_ModelProperties_json(modelName = "R Logistic", 
                           modelDescription = "R model", 
                           modelFunction = "Classification",
                           trainTable = "HMEQ",
                           algorithm = "Logistic Regression",
                           numTargetCategories = 2,
                           targetEvent = "1",
                           targetVariable = "BAD",
                           eventProbVar = "P_BAD1",
                           modeler = "jobake",
                           tool = "R",
                           toolVersion = "default",
                           path = zip_folder)

write_in_out_json(data=df_train[,-1], 
                  input=TRUE, 
                  path=zip_folder)

write_in_out_json(data=df_train_scored[,-1], 
                  input=FALSE, 
                  path=zip_folder)

write_fileMetadata_json(scoreCodeName = "scoreCode.R",
                        scoreResource = "lr_model.rds",
                        path = zip_folder)

diag = diagnosticsJson(validadedf = validData,
                       traindf = trainData,
                       testdf = testData,
                       targetEventValue = 1,
                       targetName = dm_dec_target,
                       targetPredicted = 
                       path = zip_folder)

zip_files = list.files(zip_folder, full.names = T)
zipr(zipfile=paste0(zip_folder,"/R_Logistic_Model.zip"), files=zip_files)

mod <- register_model(
  session = sess,
  file = paste0(zip_folder,"/R_Logistic_Model.zip"),
  name = "R Logistic Model",
  type = "zip",
  project = "MM_OS_Test",
  force = FALSE
)


### updload score code to table

score_tbl <- cas.table.upload(conn,
                        "C:/Users/chparr/Downloads/book1.xlsx", 
                        importOptions='xls',
                        casOut=list(name='score_test',
                                    caslib=caslib,
                                    replace=TRUE))

score_table <- cas.upload.file(conn, 
                data="C:/Users/chparr/Downloads/book1.xlsx",
                casOut=list(name="testing_score_code",
                            caslib=caslib,
                            replace=TRUE),
                importOptions=list(fileType="EXCEL",
                                   sheet="Sheet1"))


cas.table.save(s,
               table=list(name="iris",
                          caslib="casuser"), 
               name="irisCopy.sashdat", 
               replace=TRUE)

result <- cas.table.fileInfo(s,
                             path="irisCopy.sashdat")      

result



### get model information

#################
### Get Token ###
#################

library(request)

url <- paste0(hostname_model, '/SASLogon/oauth/token')

r <- POST(url=url, list=)

r <- request('POST', url,
            data='grant_type=password&username=%s&password=%s' %(username, password),
            headers={
              'Accept': 'application/json',
              'Content-Type': 'application/x-www-form-urlencoded'
            },
            auth=('sas.ec', ''),
            verify=False)
token = r.json()['access_token']


