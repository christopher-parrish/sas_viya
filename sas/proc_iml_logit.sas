data aml_bank_prep;
	set casuser.aml_bank_prep;
run;

proc iml;
/* 1. read data and form design matrix */
varNames = {'checking_only_indicator' 'prior_ctr_indicator' 'address_change_2x_indicator' 'cross_border_trx_indicator' 'in_person_contact_indicator' 'linkedin_indicator' 'marital_status_single' 'citizenship_country_risk' 'distance_to_bank' 'distance_to_employer'};
use aml_bank_prep;
read all var "ml_indicator" into y;   /* read response variable */
read all var varNames into X;   /* read explanatory variables */
close;
X = j(nrow(X), 1, 1) || X;     /* design matrix: add Intercept column */
 
/* 2. define loglikelihood function for binary logistic model */
start BinLogisticLL(b) global(X, y);
   z = X*b`;                   /* X*b, where b is a column vector */
   p = Logistic(z);            /* 1 / (1+exp(-z)) */
   LL = sum( y#log(p) + (1-y)#log(1-p) );   
   return( LL );
finish;
 
/* 3. Make initial guess and find parameters that maximize the loglikelihood */
b0 = j(1, ncol(X), 0);         /* initial guess */
opt = 1;                       /* find maximum of function */
call nlpnra(rc, b, "BinLogisticLL", b0, opt);   /* use Newton's method to find b that maximizes the LL function */
print b[c=("Intercept"||varNames) L="Parameter Estimates" F=D8.];
QUIT;