cas casauto sessopts=(caslib=casuser, metrics=true, timeout=900);

/* set macro variables */

%let in_mem_tbl = 'aml_bank_prep';
%let caslib_ref = 'casuser';
%let caslib_ref_noquote = casuser.;
%let libname_in_mem_tbl = &caslib_ref_noquote.aml_bank_prep;
%let libname_out_model = &caslib_ref_noquote.procgradboost_model;
%let out_model = "procgradboost_model";
%let libname_out_score = &caslib_ref_noquote.procgradboost_score;
%let out_score = "procgradboost_score";
%let libname_out_astore = &caslib_ref_noquote.procgradboost_astore;
%let out_astore = "procgradboost_astore";
%let libname_out_score_code = &caslib_ref_noquote.procgradboost_score_code;
%let out_score_code = "procgradboost_score_code";
%let libname_cas_out_model = &caslib_ref_noquote.casgradboost_model;
%let cas_out_model = "casgradboost_model";
%let libname_cas_out_score = &caslib_ref_noquote.casgradboost_score;
%let cas_out_score = "casgradboost_score";
%let libname_cas_out_astore = &caslib_ref_noquote.casgradboost_astore;
%let cas_out_astore = "casgradboost_astore";
%let libname_cas_out_score_code = &caslib_ref_noquote.casgradboost_score_code;
%let cas_out_score_code = "casgradboost_score_code";
%let libname_cas_out_assess = &caslib_ref_noquote.casgradboost_assess;
%let cas_out_assess = "casgradboost_assess";
%let libname_cas_out_assess_roc = &caslib_ref_noquote.casgradboost_assess;
%let cas_out_assess_roc = "casgradboost_asssess_roc";
%let target = 'ml_indicator';
%let target_noquote = ml_indicator;
%let predicted = P_ml_indicator;
%let predictedvars = "P_ml_indicator0", "P_ml_indicator1";
%let predictedtarget = 'P_ml_indicator1';
%let predictedtarget_noquote = P_ml_indicator1;
%let predictedevent_noquote = I_ml_indicator;
%let partition = analytic_partition;
%let copyvarstooutput = account_id analytic_partition;
%let inputs = {
			"checking_only_indicator", "prior_ctr_indicator", "address_change_2x_indicator",
			"cross_border_trx_indicator", "in_person_contact_indicator",
			"linkedin_indicator", "atm_deposit_indicator", "trx_10ksum_indicator",
			"common_merchant_indicator", "direct_deposit_indicator", "marital_status_single",
			"primary_transfer_cash", "citizenship_country_risk", "occupation_risk", 
			"credit_score", "distance_to_bank", "distance_to_employer", 
			"income", "num_acctbal_chgs_gt2000", "num_transactions"};
%let inputs_noquoteorcomma = 
			checking_only_indicator prior_ctr_indicator address_change_2x_indicator
			cross_border_trx_indicator in_person_contact_indicator
			linkedin_indicator atm_deposit_indicator trx_10ksum_indicator
			common_merchant_indicator direct_deposit_indicator marital_status_single
			primary_transfer_cash citizenship_country_risk occupation_risk 
			credit_score distance_to_bank distance_to_employer 
			income num_acctbal_chgs_gt2000 num_transactions;
%let bias_var = "cross_border_trx_indicator";
%let partial_dep_var = "credit_score";


/* load training table into memory */

proc cas;
		table.tableExists result=code /
			caslib=&caslib_ref, name=&in_mem_tbl;
			if code['exists'] <= 0 then do;
				table.loadTable /
					caslib=&caslib_ref,
					path=cats(&in_mem_tbl,'.sashdat'),
					casout={caslib=&caslib_ref,
						name=&in_mem_tbl,
						promote=TRUE};
			end;
		table.columnInfo result=col_list /
			table={caslib=&caslib_ref, name=&in_mem_tbl};
			describe col_list;
			print col_list.ColumnInfo[,'Column'];
run;


/* proc gradboost procedure */

proc gradboost data=&libname_in_mem_tbl
		outmodel=&libname_out_model
		seed=12345
     	ntrees=100
		learningrate=0.1 
		samplingrate=0.5 
		lasso=0 
		ridge=1 
		maxbranch=2
		maxdepth=5
		minleafsize=5
		assignmissing=USEINSEARCH 
		minuseinsearch=1
		nomsearch(maxcategories=128)
		numBin=50 
		binmethod=QUANTILE
		earlystop(tolerance=0 stagnation=5 minimum=NO metric=MCR);
		*autotune;
		*code file=;
		*crossvalidation;
		*weight;
		id &target_noquote;
		partition rolevar=&partition (train='1' validate='0' test='2');
		input &inputs_noquoteorcomma / level=interval; 		
		target &target_noquote / level=nominal;
		output out=&libname_out_score 
					copyvars=(&copyvarstooutput);
		savestate rstore=&libname_out_astore;
		ods output
			VariableImportance=work.varimportance
			Fitstatistics=work.fitstatistics;
run;

proc cas;

	decisionTree.gbtreeCode /
		modelTable={caslib=&caslib_ref, name=&out_model}
		code={casOut={caslib=&caslib_ref, name=&out_score_code, 
						replace=True, promote=False}}
	;
run;

proc freqtab data=&libname_out_score;
	by analytic_partition;
	tables &target_noquote*&predictedevent_noquote / nopercent;
run;

proc npar1way data=&libname_out_score edf;
	class &target_noquote;
	var &predictedtarget_noquote;
run;


/* gbtreeTrain action */

proc cas;
	decisionTree.gbtreeTrain /
		table={caslib=&caslib_ref, name=&in_mem_tbl}
		target=&target
		inputs=&inputs
		encodeName=TRUE
		nominals={&target}
		m=20
		seed=12345
		nTree=100
		learningRate=0.1
		subSampleRate=0.5
		lasso=0
		ridge=1
		distribution="binary"
		maxBranch=2
		maxLevel=5
		leafSize=5
		missing="useinsearch"
		minUseInSearch=1
		nBins=50
		quantileBin=TRUE
		earlyStop={metric="MCR", stagnation=5, tolerance=0, minimum=FALSE,
					threshold=0, thresholdIter=0}
		casOut={caslib=&caslib_ref, name=&cas_out_model, replace=True}
		saveState={caslib=&caslib_ref, name=&cas_out_astore, replace=True}
    	;
run;

proc cas;

	decisionTree.gbtreeCode /
		modelTable={name=&cas_out_model}
		code={casOut={caslib=&caslib_ref, name=&cas_out_score_code, 
						replace=True, promote=False}}
	;
run;

proc cas;

	decisionTree.gbtreeScore /
		modelTable={name=&cas_out_model}
		table={caslib=&caslib_ref, name=&in_mem_tbl}
		casOut={caslib=&caslib_ref, name=&cas_out_score, replace=True}
		copyVars={&target}
		encodeName=TRUE
		assessOneRow=TRUE
	;
run;

proc cas;
	percentile.assess /
		table={caslib=&caslib_ref, name=&cas_out_score}
		event="1"
		response=&target
		inputs=&predictedtarget
		cutStep=0.001
		casOut={caslib=&caslib_ref, name=&cas_out_assess, replace=True}
	;
run;

data test (where=(_KS_=1));
	set &libname_cas_out_assess_roc;
run;

/* _KS2_ is the highest KS given cutSteps */
proc print data=test;
run;


/* model interpretability - single obs */

data public.shapley_obs;
	set public.aml_bank_prep;
	if account_id = 5;
run;

data public.shapley_train;
	set public.aml_bank_prep;
run;

proc cas;

   explainModel.shapleyExplainer / 
		table={caslib=&caslib_ref, name="shapley_train"}
		query={caslib=&caslib_ref, name="shapley_obs"}
		modelTable={caslib=&caslib_ref, name="procgradboost_astore"}
		modelTableType="astore"
		predictedTarget=&predictedtarget
		inputs=&inputs
		depth=1
		outputTables={includeAll=TRUE};
run;

quit;

/* model interpretability - multiple obs */

%let n_account_id=100;

data public.shapley_train;
	set public.aml_bank_prep;
run;

%macro shapley;

%do i=1 %to &n_account_id.;
	data public.shapley_temp;
		set public.aml_bank_prep;
		if account_id = &i.;
	run;
	proc cas;
   		explainModel.shapleyExplainer / 
			table={caslib=&caslib_ref, name="shapley_train"}
			query={caslib=&caslib_ref, name="shapley_temp"}
			modelTable={caslib=&caslib_ref, name="procgradboost_astore"}
			modelTableType="astore"
			predictedTarget=&predictedtarget
			inputs=&inputs
			depth=1
			outputTables={includeAll=TRUE}
			;
		transpose.transpose /
			table={caslib=&caslib_ref, name="shapleyvalues"}
   			id={"variable"}
   			casOut={caslib=&caslib_ref, name="shapley_transpose", replace=true}
			;
   	run;
	quit;
	%if &i. = 1 %then %do;
		data public.shapley_a;
			set public.shapley_transpose;
		run;
		data public.shapley_b;
			set public.shapleyvalues;
		run;
		%end;
	%else %do;
		data public.shapley_a(append=force);
			set public.shapley_transpose;
		run;
		data public.shapley_b(append=force);
			set public.shapleyvalues;
		run;
		%end;
%end;

%mend shapley;
%shapley;

proc casutil;
	promote casdata='shapley_b' incaslib=&caslib_ref 
			casout='shapley_b' outcaslib=&caslib_ref;
run;


/* charting HyperSHAP values */

data public.shapley_avg;
	set public.shapley_b;
	ShapleyValue = abs(ShapleyValue);
run;

proc sgplot data=public.shapley_avg;
	title "Average Absolute HyperSHAP Values";
  	hbar Variable /
    response=ShapleyValue
    stat=mean
	categoryorder=respdesc;
run;
title;

proc sgplot data=public.shapley_b;
	title "HypeSHAP Values by Attribute";
	heatmap x=ShapleyValue y=Variable /
		colormodel=(blue yellow red)
		nxbins=25
   		;
run;
title;


/* k-fold results */

proc cas;
	sampling.kfold /
    	table={caslib=&caslib_ref, name=&in_mem_tbl}
    	k=10
    	seed=12345
		output={casOut={caslib=&caslib_ref, name="gradboost_kfold", replace=true},
				copyVars='ALL',
				foldname='_fold_'}
	;
run;

%macro kfold;

%do i=1 %to 10;

		proc cas;
    	decisionTree.gbtreeTrain /
			table={caslib=&caslib_ref, name='gradboost_kfold', where="_fold_ = " || &i.}
			target=&target
			inputs=&inputs
			encodeName=TRUE
			nominals={&target}
			m=20
			seed=12345
			nTree=100
			learningRate=0.1
			subSampleRate=0.5
			lasso=0
			ridge=1
			distribution="binary"
			maxBranch=2
			maxLevel=5
			leafSize=5
			missing="useinsearch"
			minUseInSearch=1
			nBins=50
			quantileBin=TRUE
			earlyStop={metric="MCR", stagnation=5, tolerance=0, minimum=FALSE,
					threshold=0, thresholdIter=0}
			casOut={caslib=&caslib_ref, name="casgradboost_temp", replace=True}
    		;

		decisionTree.gbtreeScore /
			modelTable={name="casgradboost_temp"}
			table={caslib=&caslib_ref, name='gradboost_kfold', where="_fold_ = " || &i.}
			casOut={caslib=&caslib_ref, name='casgradboost_temp_scored', replace=True}
			copyVars={&target}
			encodeName=TRUE
			assessOneRow=TRUE
			;
  		run;

		data public.temp;
			set public.casgradboost_temp_scored;
			I_indicator = round(P_ml_indicator,1);
			diff = abs(ml_indicator - I_indicator);
		run;

		proc cas;

		freqTab.freqTab /
      		table={caslib=&caslib_ref, name='temp'},
			tabulate={'I_indicator'},
			outputTables={includeAll=TRUE}
		;
		run;

		data public.onewayfreqs;
			set public.onewayfreqs;
			_fold_ = &i.;
		run;

		%if &i. = 1 %then %do;
			data public.kfold_output;
				set public.onewayfreqs;
			run;
		%end;
		%else %do;
			data public.kfold_output(append=force);
				set public.onewayfreqs;
			run;
		%end;

%end;

%mend kfold;
%kfold;

proc casutil;
	promote casdata='kfold_output' incaslib=&caslib_ref 
			casout='kfold_output' outcaslib=&caslib_ref;
run;

/* oversampling */

proc cas;
	sampling.oversample /
		table={caslib=&caslib_ref, name=&in_mem_tbl, groupby={&target}}
		event="1" samppctevt=90 eventprop=0.5 partind="false" seed=12345
	    output={casOut={caslib=&caslib_ref, name=&in_mem_tbl || '_oversample', replace=True},
			copyVars='ALL', freqName='freq'}
	;
run;

/* re-sampling */

proc surveyselect data=public.aml_bank_prep
	seed=12345
	method=balbootstrap
	reps=10
	outhits
	out=public.aml_bank_prep_boot;
	samplingunit ml_indicator;
	strata occupation_risk;
run;

*proc surveyselect data=Customers method=seq n=(8 12 20 10)
                  reps=4 seed=40070 ranuni out=SampleRep;

/* bias assessment */

proc astore;
	describe rstore=public.casgradboost_astore;
run;

proc cas;
	fairAITools.assessBias /
		table = {caslib=&caslib_ref, name=&in_mem_tbl},
		modelTable = "casgradboost_astore",
		modelTableType = "ASTORE",
		response = &target,
		predictedVariables = {"P_ml_indicator0", "P_ml_indicator1"},
		responseLevels={"0", "1"},
		sensitiveVariable = &bias_var,
		scoredTable='bias_values'
	;
run;