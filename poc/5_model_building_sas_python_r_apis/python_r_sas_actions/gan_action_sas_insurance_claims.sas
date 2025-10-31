cas casauto sessopts=(caslib=CASUSER, metrics=true, timeout=900);

proc cas;
     	 generativeAdversarialNet.tabularGanTrain result = r /
         table           = {caslib = 'casuser', name = "insurance_claims_auto",
                            vars = 
							{"claim_indicator",
							 "new_driver",
							 "bundle",
							 "ever_deferred_payment",
							 "no_fault_state",
							 "channel_type"}},
         nominals        = {"channel_type"},
		 gmmOptions		 = {alpha = 1, maxClusters = 10, seed = 42,
							inference = {covariance = 'DIAGONAL',
										 maxVbIter = 30,
										 threshold = 0.01}},
         gpu             = {useGPU = False, device = 0},
         optimizerAe     = {method = "ADAM", numEpochs = 3},
         optimizerGan    = {method = "ADAM", numEpochs = 5},
         seed            = 12345,
         scoreSeed       = 0,
         numSamples      = 5,
         saveState       = {name = 'insurance_claims_synth_astore', replace = True},
         casOut          = {name = 'insurance_claims_synth_out', replace = True};
     print r;
 run;
 quit;

 proc print data = casuser.insurance_claims_synth_out;
 run;

proc sgplot data=casuser.insurance_claims_auto;
	title "original";
	histogram new_driver;
	density new_driver;
	density new_driver / type=kernel;
	keylegend / location = inside position = topright;
run;
title;

proc sgplot data=casuser.insurance_claims_synth_out;
	title "synthetic";
	histogram new_driver;
	density new_driver;
	density new_driver / type=kernel;
	keylegend / location = inside position = topright;
run;
title;

proc sgplot data=casuser.insurance_claims_auto;
	title "original";
	density new_driver / type=kernel;
run;
title;

proc sgplot data=casuser.insurance_claims_synth_out;
	title "synthetic";
	density new_driver / type=kernel;
run;
title;

/* the binary variable new_driver is treated as an interval var.
  if you change it to nominal, such as income, you'll get a smoother distributions.
  interesting to note that the original var was 80/20 (0/1) split while the synthetic data 
  are 60/40, so it appears that the model is balancing out the sample.  
  for interval vars, like income, the distribution is similar, but it does come up with 
  some negative values.  i'll have to check to see if there is a way to parameterize 
  these boundaries. */

proc sgplot data=casuser.insurance_claims_auto;
	title "original";
	density income / type=kernel;
run;
title;

proc sgplot data=casuser.insurance_claims_synth_out;
	title "synthetic";
	density income / type=kernel;
run;
title;

/*                             vars = 
							{"claim_indicator",
							 "new_driver",
							 "bundle",
							 "ever_deferred_payment",
							 "no_fault_state",
							 "marital_status",
							 "vehicle_history",
							 "num_auto_drivers",
							 "num_claims",
							 "deductible",
							 "liability_limit",
							 "num_prior_claims",
							 "claim_amt",
							 "income",
							 "policy_age_mos",
							 "owner_age",
							 "vehicle_age",
							 "auto_points",
							 "credit_score",
							 "claim_indicator_type",
							 "marital_status_type",
							 "vehicle_history_type",
							 "channel_type"}},
         nominals        = {"claim_indicator_type",
							"marital_status_type",
							"vehicle_history_type",
							"channel_type"},
*/