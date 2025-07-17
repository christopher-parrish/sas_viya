#################################################################
###			Train Logit Model in SAS Viya's Build Models App		  ###
#################################################################

########################
### Model Parameters ### 
########################

### import r packages
library(dplyr)

### model arguments
link <- 'logit'

### model manager information
model_name <- 'logit_r_amlbank'
project_name <- 'Anti-Money Laundering'
description <- 'Logit R'
model_type <- 'logisitic_regression'

### define macro variables for model
dm_dec_target <- 'ml_indicator'
dm_partitionvar <- 'analytic_partition'
create_new_partition <- 'no' # 'yes'/'no'
dm_key <- 'account_id' 
dm_classtarget_level <- list('0', '1')
dm_partition_val <- list('dm_partition_validate_val'=0, 'dm_partition_train_val'=1, 'dm_partition_test_val'=2)
dm_partition_perc <- list('dm_partition_validate_perc'=0.3, 'dm_partition_train_perc'=0.6, 'dm_partition_test_perc'=0.1)
avg_prob <- mean(dm_inputdf[[dm_dec_target]], na.rm = TRUE)

### create list of model variables
### if already rejected in project, do not include in the rejected_predictors list
### alternatively, state keep_predictors and rejected_predictors will be calculated
keep_predictors <- list(
  'checking_only_indicator',
  'prior_ctr_indicator',
  'address_change_2x_indicator',
  'cross_border_trx_indicator',
  'in_person_contact_indicator',
  'linkedin_indicator',
  'trx_10ksum_indicator', 
  'common_merchant_indicator', 
  'direct_deposit_indicator',
  'marital_status_single',
  'primary_transfer_cash', 
  'credit_score', 
  'num_transactions'
)

##############################
### Final Modeling Columns ###
##############################

### remove added index column in model studio
#dm_inputdf <- subset(dm_inputdf, select=-c('_dmIndex_'))

### create list of model variables
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
###	Data Split ###
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
###	Training Code	###
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
