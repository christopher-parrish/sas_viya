library(float)

option_value <- function(notional, vol, strike_price, spot_price, time_to_mat, risk_free_rate) {
  #output: option_val
  d1 <- (log(spot_price/strike_price) + (risk_free_rate+vol**2/2)*time_to_mat)/(vol*sqrt(time_to_mat))
  d2 <- d1 - vol*sqrt(time_to_mat)
  option_price <- pnorm(d1)*spot_price-pnorm(d2)*strike_price*exp(-risk_free_rate*time_to_mat)
  option_val <- as.float(notional * option_price)
  output_list <- list('option_val'= option_val)
  return (output_list)}
