cas casauto sessopts=(caslib=CASUSER, metrics=true, timeout=900);

/* Define the runloop macro */

%macro runloop;
     /* Specify all interval input variables*/
     %let names = credit_score
					amount;
     /* Loop over all variables that need centroids generation */
     %do i=1 %to %sysfunc(countw(&names));
         %let name&i = %scan(&names, &i, %str( ));
         /* Call the GMM action to cluster each variable */
         proc cas ;
             action nonParametricBayes.gmm result=R/
                 table       = {caslib='public', name="financial_services"},
                 inputs      = {"&&name&i"},
                 seed        = 1234567890,
                 maxClusters = 10,
                 alpha       = 1,
                 infer       = {method="VB",
                                maxVbIter =30,
                                covariance="diagonal",
                                threshold=0.01},
                 output      = {casOut={name='Score', replace=true},
                                copyVars='ALL'},
                 display     = {names={ "ClusterInfo"}}
                ;
             run;
             saveresult R.ClusterInfo replace dataset=work.weights&i;
         run;
         quit;

         /* Save variable name, weights, mean,     */
         /* and standard deviation of each cluster */
         data  weights&i;
             varname = "&&name&i";
             set  weights&i(rename=(&&name&i.._Mean=Mean
                                    &&name&i.._Variance=Var));
             /* Calculate standard deviation from variance*/
             std = sqrt(Var);
             drop Var;
         run;

         /* Construct centroids table from saved weights */
         %if &i=1 %then %do;
             data centroids;
             set weights&i;
             run;
         %end;
         %else %do;
             data centroids;
             set centroids weights&i;
             run;
         %end;
     %end;
 %mend;

 /* Run the runloop macro to generate the centroids table */

 %runloop;

/* The following DATA step uploads the centroids table to the CAS session */

data casuser.centroids;
   set centroids;
run;

/* The following PROC CAS statements use the tabularGanTrain action to train a tabular GAN model and save the trained model in an analytic store. The PROC PRINT statement prints the generated samples. */

proc cas;
     	 generativeAdversarialNet.tabularGanTrain result = r /
         table           = {caslib = 'public', name = "financial_services",
                            vars = {'credit_score',	'amount'}},
         centroidsTable  = {caslib = "casuser", name = "centroids"},
         nominals        = {"account_id"},
         gpu             = {useGPU = False, device = 0},
         optimizerAe     = {method = "ADAM", numEpochs = 3},
         optimizerGan    = {method = "ADAM", numEpochs = 5},
         seed            = 12345,
         scoreSeed       = 0,
         numSamples      = 5,
		 miniBatchSize	 = 100,
		 packSize		 = 4,
         saveState       = {name = 'financial_services_astore', replace = True},
         casOut          = {name = 'financial_services_out', replace = True};
     print r;
 run;
 quit;

 proc print data = casuser.financial_services_out;
 run;