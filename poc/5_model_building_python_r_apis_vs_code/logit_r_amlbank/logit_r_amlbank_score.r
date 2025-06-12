library("stats")

scoreFunction <- function(num_transactions, credit_score, marital_status_married, marital_status_divorced, checking_only_indicator, prior_ctr_indicator, address_change_2x_indicator, cross_border_trx_indicator, in_person_contact_indicator, linkedin_indicator, trx_10ksum_indicator, common_merchant_indicator, direct_deposit_indicator, primary_transfer_check, primary_transfer_wire)
{
  #output: EM_CLASSIFICATION, EM_EVENTPROBABILITY, EM_PROBABILITY, I_ml_indicator, P_ml_indicator1, P_ml_indicator0
  
  if (!exists("sasctlRmodel"))
  {
    assign("sasctlRmodel", readRDS(file = paste(rdsPath, "logit_r_amlbank.rds", sep = "")), envir = .GlobalEnv)
  }
  
  data <- data.frame(num_transactions  =  num_transactions,
                     credit_score  =  credit_score,
                     marital_status_married  =  marital_status_married,
                     marital_status_divorced  =  marital_status_divorced,
                     checking_only_indicator  =  checking_only_indicator,
                     prior_ctr_indicator  =  prior_ctr_indicator,
                     address_change_2x_indicator  =  address_change_2x_indicator,
                     cross_border_trx_indicator  =  cross_border_trx_indicator,
                     in_person_contact_indicator  =  in_person_contact_indicator,
                     linkedin_indicator  =  linkedin_indicator,
                     trx_10ksum_indicator  =  trx_10ksum_indicator,
                     common_merchant_indicator  =  common_merchant_indicator,
                     direct_deposit_indicator  =  direct_deposit_indicator,
                     primary_transfer_check  =  primary_transfer_check,
                     primary_transfer_wire  =  primary_transfer_wire)
  
  predictions <- predict(sasctlRmodel, newdata = data, type = "response")

  P_ml_indicator1 <- predict(sasctlRmodel, newdata = data, type = "response")
  P_ml_indicator0 <- 1 - P_ml_indicator1

  output_list <- list(EM_CLASSIFICATION = ifelse(P_ml_indicator1 >= 0.5, 1, 0), 
                      EM_EVENTPROBABILITY = P_ml_indicator1,
                      EM_PROBABILITY = ifelse(P_ml_indicator1 >= 0.5, P_ml_indicator1, P_ml_indicator0),
                      I_ml_indicator = ifelse(P_ml_indicator1 >= 0.5, 1, 0),
                      P_ml_indicator1 = P_ml_indicator1,
                      P_ml_indicator0 = P_ml_indicator0)

  return(output_list)
}
