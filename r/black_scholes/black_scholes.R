

###################
### Credentials ###
###################

output_dir <- 'C:XXX'

###################
### Environment ###
###################

library(swat)
library(sys)

wd <- 'C:XXX'
setwd(wd)
conn <- swat::CAS(hostname = 'https://XXX/cas-shared-default-http/',
                  port = 443, 
                  username = 'XXX',
                  password = 'XXX',
                  protocol='http'
                  )
print(conn)
print(cas.builtins.serverStatus(conn))

### caslib and table to use in scoring
caslib <- 'Public'
in_mem_tbl <- 'black_scholes_score'

##########################
### Create Score Table ###
##########################

### model manager information
model_name <- 'black_scholes_r'
score_code_name <- 'black_scholes_rScore.r'
project_name <- 'Black_Scholes_Option'
description <- 'UDF'
model_type <- 'UDF'
metadata_output_dir <- 'outputs'
dm_dec_target <- 'option_val'

### create table to score model
data_x <- list(200, .5, 15, 5, 1, 1)
columns_x <- c('notional', 'vol', 'strike_price', 'spot_price', 'time_to_mat', 'risk_free_rate')
X <- data.frame(data_x)
names(X) <- columns_x

### load table in-memory if not already exists in-memory ###
if (cas.table.tableExists(conn, caslib=caslib, name=in_mem_tbl)<=0) {
  cas.upload(conn, data=X, casout=list(name=in_mem_tbl, caslib=caslib, promote=TRUE))}

#########################
### Create json Files ###
#########################

library(jsonlite)

output_path <- file.path(output_dir, metadata_output_dir, model_name)
if (file.exists(output_path)) {
  unlink(output_path, recursive=TRUE) }

dir.create(output_path)

fileMetadata <- data.frame(
  'role' = c('inputVariables', 'outputVariables', 'score'),
  'name' = c('inputVar.json', 'outputVar.json', 'black_scholes_rScore.r'))
write_json(fileMetadata, path=file.path(output_path, 'fileMetadata.json'), pretty=TRUE, auto_unbox=TRUE)

ModelProperties <- list('name' = model_name,
                        'description' = description,
                        'function' = 'prediction',
                        'scoreCodeType' = 'r',
                        'trainTable' = ' ',
                        'trainCodeType' = 'r',
                        'algorithm' = model_type,
                        'targetVariable' = "",
                        'targetEvent' = "",
                        'targetLevel' = "",
                        'eventProbVar' = dm_dec_target,
                        'modeler' = Sys.info()['user'],
                        'tool' = R.version$language,
                        'toolVersion' = R.version.string)

write_json(ModelProperties, path=file.path(output_path, 'ModelProperties.json'), pretty=TRUE, auto_unbox=TRUE)

inputVar <- data.frame('name' = character(), 'length' = integer(), 'type' = character(), 'level' = character())
for (var in columns_x)
{
  vType = 'decimal'
  vList <- data.frame('name' = var, 'length' = 8, 'type' = vType, 'level' = 'interval')
  inputVar <- rbind(inputVar, vList, stringsAsFactors = FALSE)
}
write_json(inputVar, path=file.path(output_path, 'inputVar.json'), pretty=TRUE, auto_unbox=TRUE)

outputVar <- data.frame(
  'name' = c(dm_dec_target),
  'length' = c(8),
  'type' = c('decimal'),
  'level' = c('interval'))
write_json(outputVar, path=file.path(output_path, 'outputVar.json'), pretty=TRUE, auto_unbox=TRUE)

### copy score code over to directory that will be zipped
file.copy(file.path(output_dir, score_code_name), file.path(output_path, score_code_name))


##########################
### Write RDs/RDA File ###
##########################

### this would be where the model is 'pickled'

objs <- unlist(data_x)

saveRDS(objs, file=file.path(output_path, paste(model_name, ".rds", sep ="")))

rdsPath <- 'C:/Users/chparr/OneDrive - SAS/r/outputs/black_scholes_r/black_scholes_r.rds'
readRDS(file = rdsPath)

#################
### Zip Files ###
#################

setwd(output_path)
files2zip <- dir(output_path)
zippath <- file.path(output_path, paste(model_name, ".zip", sep =""))
zip::zip(zipfile=zippath, files=files2zip)




