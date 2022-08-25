
##################
### Score Code ###
##################


rscoreFunction <- function(checking_only_indicator, 
                           prior_ctr_indicator,
                           address_change_2x_indicator,
                           cross_border_trx_indicator,
                           in_person_contact_indicator,
                           linkedin_indicator,
                           trx_10ksum_indicator,
                           common_merchant_indicator,
                           direct_deposit_indicator,
                           marital_status,
                           primary_transfer_cat,
                           credit_score,
                           num_transactions)
{
  #output: EM_EVENTPROBABILITY, EM_CLASSIFICATION
  
  if (!exists("dm_model"))
  {
    assign("dm_model", readRDS(file = paste(rdsPath, 'logit_r.rds', sep = '')), envir = .GlobalEnv)
  }
  
  threshPredProb <- 0.0500629282617816
  
  input_array <- data.frame("checking_only_indicator" = checking_only_indicator,
                            "prior_ctr_indicator" = prior_ctr_indicator,
                            "address_change_2x_indicator" = address_change_2x_indicator,
                            "cross_border_trx_indicator" = cross_border_trx_indicator,
                            "in_person_contact_indicator" = in_person_contact_indicator,
                            "linkedin_indicator" = linkedin_indicator,
                            "trx_10ksum_indicator" = trx_10ksum_indicator,
                            "common_merchant_indicator" = common_merchant_indicator,
                            "direct_deposit_indicator" = direct_deposit_indicator,
                            "marital_status" = marital_status,
                            "primary_transfer_cat" = primary_transfer_cat,
                            "credit_score" = credit_score,
                            "num_transactions" = num_transactions)
  
  predProb <- predict(dm_model, newdata=input_array, type="response", na.action = na.omit)
  
  if (!is.na(predProb[])) {
    EM_EVENTPROBABILITY <- predProb[]
    EM_CLASSIFICATION <- ifelse(EM_EVENTPROBABILITY >= threshPredProb, '1', '0')
  } else {
    EM_EVENTPROBABILITY <- NA
    EM_CLASSIFICATION <- ' '
  }

  output_list <- list('EM_EVENTPROBABILITY' = EM_EVENTPROBABILITY, 'EM_CLASSIFICATION' = EM_CLASSIFICATION)
  return(output_list)
}

