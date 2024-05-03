cas casauto sessopts=(caslib=casuser, metrics=true, timeout=1800);
libname cp cas caslib=casuser;
caslib _all_ assign;

/***********************************
* PROC IML Procedure - Train Model *
************************************/

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
run;
quit;

/*******************************
* IML Action Set - Train Model *
********************************/

proc cas;
loadactionset "iml";
source aml_logit;
	/* 1. read data and form design matrix */
	*** note that MatrixCreateFromCAS function will keep/drop in the
		order of the table, not in the order of the column names.
		it is recommended to order column names as they occur in the table ***;
	keep_x = 'KEEP=
				marital_status_single				
				checking_only_indicator
				prior_ctr_indicator
				address_change_2x_indicator
				cross_border_trx_indicator
				in_person_contact_indicator
				linkedin_indicator
				citizenship_country_risk
				distance_to_employer
				distance_to_bank';
	keep_y = 'KEEP= ml_indicator';
	varNames = {'marital_status_single' 'checking_only_indicator' 'prior_ctr_indicator' 'address_change_2x_indicator' 'cross_border_trx_indicator' 'in_person_contact_indicator' 'linkedin_indicator' 'citizenship_country_risk' 'distance_to_employer' 'distance_to_bank'};
	X = MatrixCreateFromCAS('casuser', 'aml_bank_prep', keep_x);
	y = MatrixCreateFromCAS('casuser', 'aml_bank_prep', keep_y);
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
	opt = {-1};                    /* find maximum of function */
	call nlpsolve(rc, b, "BinLogisticLL", b0) OPT=opt;   /* use Newton's method to find b that maximizes the LL function */
	print b[c=("Intercept"||varNames) L="Parameter Estimates" F=D8.];
endsource;
iml / code = aml_logit nthreads=8;
run;
quit;

/************************************
* IML Action Set - Scoring Function *
*************************************/

proc cas;
loadactionset "iml";
source score_model;
   /* This function scores one observation at a time */
   start scoreFunc(beta, xn);
      P_ml_indicator1 = exp(beta[1] + beta[2]*xn[1] + beta[3]*xn[2] + beta[4]*xn[3] +
							+ beta[5]*xn[4] + beta[6]*xn[5] + beta[7]*xn[6] +
							+ beta[8]*xn[7] + beta[9]*xn[8] + beta[10]*xn[9] +
							+ beta[11]*xn[10]);
      return P_ml_indicator1;
   finish;

   beta={-6.5838, 2.4328, 1.3476, 1.2040, 1.7209, 1.6382, -1.9480, -2.0010, 0.4066, 0.0597, -0.00019};
   varNames = {'marital_status_single' 'checking_only_indicator' 'prior_ctr_indicator' 'address_change_2x_indicator' 'cross_border_trx_indicator' 'in_person_contact_indicator' 'linkedin_indicator' 'citizenship_country_risk' 'distance_to_employer' 'distance_to_bank'};
   copyVars = varNames || {'ml_indicator'};

   rc = Score('scoreFunc',    /* name of the scoring function */
              beta,          /* scoring constants */
              varNames,        /* input variables */
              'P_ml_indicator1',        /* output variables */
              'aml_bank_prep',        /* input CAS table */
              'aml_bank_score',    /* output CAS table */
              1,             /* pass 1 row at a time */
              copyVars);     /* copy vars from input to output */

	*** create astore ***;
	analytics_store = astore('aml_bank_astore', 'scoreFunc', beta, varNames, 'P_ml_indicator1');
endsource;
iml / code=score_model, nthreads=8;
run;
quit;

/***************
* Scored Table *
****************/

proc print data=casuser.aml_bank_score(obs=5);
   var P_ml_indicator1
		marital_status_single				
		checking_only_indicator
		prior_ctr_indicator
		address_change_2x_indicator
		cross_border_trx_indicator
		in_person_contact_indicator
		linkedin_indicator
		citizenship_country_risk
		distance_to_employer
		distance_to_bank;
run;

/************************
* Score Table w. Astore *
*************************/

proc astore;
	score rstore=casuser.aml_bank_astore data=casuser.aml_bank_prep
					out=casuser.aml_bank_score_astore copyVars=(_ALL_);
run;

proc print data=casuser.aml_bank_score_astore(obs=5);
   var P_ml_indicator1
		marital_status_single				
		checking_only_indicator
		prior_ctr_indicator
		address_change_2x_indicator
		cross_border_trx_indicator
		in_person_contact_indicator
		linkedin_indicator
		citizenship_country_risk
		distance_to_employer
		distance_to_bank;
run;