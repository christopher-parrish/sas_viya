/* 1. read data and form design matrix */
/* 2. define loglikelihood function for binary logistic model */
/* 3. Make initial guess and find parameters that maximize the loglikelihood */


proc iml;
varNames = {'checking_only_indicator' 'prior_ctr_indicator' 'address_change_2x_indicator' 'cross_border_trx_indicator' 'in_person_contact_indicator' 'linkedin_indicator' 'marital_status_single' 'citizenship_country_risk' 'distance_to_bank' 'distance_to_employer'};
use &dm_data;
read all var "ml_indicator" into y;
read all var varNames into X;
close;
X = j(nrow(X), 1, 1) || X;
 
start BinLogisticLL(b) global(X, y);
   z = X*b`;
   p = Logistic(z);
   LL = sum( y#log(p) + (1-y)#log(1-p) );   
   return( LL );
finish;
 
b0 = j(1, ncol(X), 0);
opt = 1;
call nlpnra(rc, b, "BinLogisticLL", b0, opt);
print b[c=("Intercept"||varNames) L="Parameter Estimates" F=D8.];
run;
quit;