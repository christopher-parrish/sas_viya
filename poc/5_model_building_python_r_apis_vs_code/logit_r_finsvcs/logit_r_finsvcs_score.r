library("stats")

scoreFunction <- function(gender, at_current_job_1_year, num_dependents, age, debt_to_income, num_transactions, credit_history_mos, credit_score, amount, net_worth, job_in_education, job_in_hospitality)
{
  #output: EM_CLASSIFICATION, EM_EVENTPROBABILITY, EM_PROBABILITY, I_event_indicator, P_event_indicator1, P_event_indicator0
  
  if (!exists("sasctlRmodel"))
  {
    assign("sasctlRmodel", readRDS(file = paste(rdsPath, "logit_r_finsvcs.rds", sep = "")), envir = .GlobalEnv)
  }
  
  data <- data.frame(gender  =  gender,
                     at_current_job_1_year  =  at_current_job_1_year,
                     num_dependents  =  num_dependents,
                     age  =  age,
                     debt_to_income  =  debt_to_income,
                     num_transactions  =  num_transactions,
                     credit_history_mos  =  credit_history_mos,
                     credit_score  =  credit_score,
                     amount  =  amount,
                     net_worth  =  net_worth,
                     job_in_education  =  job_in_education,
                     job_in_hospitality  =  job_in_hospitality)
  
  predictions <- predict(sasctlRmodel, newdata = data, type = "response")

  P_event_indicator1 <- predict(sasctlRmodel, newdata = data, type = "response")
  P_event_indicator0 <- 1 - P_event_indicator1

  output_list <- list(EM_CLASSIFICATION = event_indicator, 
                      EM_EVENTPROBABILITY = P_event_indicator1,
                      EM_PROBABILITY = ifelse(P_event_indicator1 >= 0.5, P_event_indicator1, P_event_indicator0),
                      I_event_indicator = ifelse(P_event_indicator1 >= 0.5, 1, 0),
                      P_event_indicator1 = P_event_indicator1,
                      P_event_indicator0 = P_event_indicator0)

  return(output_list)
}
