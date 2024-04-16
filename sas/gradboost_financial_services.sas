cas casauto sessopts=(caslib=PUBLIC, metrics=true, timeout=900);

/* set macro variables */

%let in_mem_tbl = 'financial_services_prep';
%let caslib_ref = 'casuser';
%let libname_in_mem_tbl = casuser.financial_services_prep;
%let target = 'event_indicator';
%let target_noquote = event_indicator;
%let predicted = P_event_indicator;
%let predictedvars = "P_event_indicator0", "P_event_indicator1";
%let predictedtarget = 'P_event_indicator1';
%let predictedtarget_noquote = P_event_indicator1;
%let predictedevent = I_event_indicator;
%let partition = analytic_partition;
%let copyvarstooutput = account_id analytic_partition;
%let inputs = {
			"age", "amount", "at_current_job_1_year", "business_owner", "citizenship",
			"credit_history_mos", "credit_score", "debt_to_income",
			"ever_missed_obligation", "gender", "homeowner", "job_industry",
			"marital_status", "net_worth", "num_dependents",
			"num_transactions", "region", "smoker", "uses_direct_deposit",
			"years_at_residence"};
%let inputs_noquoteorcomma = 
			age amount at_current_job_1_year business_owner citizenship
			credit_history_mos credit_score debt_to_income
			ever_missed_obligation gender homeowner job_industry
			marital_status net_worth num_dependents
			num_transactions region smoker uses_direct_deposit
			years_at_residence;
%let bias_var = "gender";
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
		outmodel=casuser.procgradboost_model
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
		output out=casuser.gradboost_score 
					copyvars=(&copyvarstooutput);
		savestate rstore=casuser.procgradboost_astore;
		ods output
			VariableImportance=work.varimportance
			Fitstatistics=work.fitstatistics;
run;

proc cas;

	decisionTree.gbtreeCode /
		modelTable={name="procgradboost_model"}
		code={casOut={caslib=&caslib_ref, name='procgradboost_scorecode', 
						replace=True, promote=False}}
	;
run;

proc freqtab data=public.gradboost_score;
	by &partition;
	tables &target_noquote*&predictedevent / nopercent;
run;

proc npar1way data=public.gradboost_score edf;
	class &target;
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
		casOut={caslib=&caslib_ref, name="casgradboost_model", replace=True}
		saveState={caslib=&caslib_ref, name="casgradboost_astore", replace=True}
    	;
run;

proc cas;

	decisionTree.gbtreeCode /
		modelTable={name="casgradboost_model"}
		code={casOut={caslib=&caslib_ref, name='casgradboost_scorecode', 
						replace=True, promote=False}}
	;
run;

proc cas;

	decisionTree.gbtreeScore /
		modelTable={name="casgradboost_model"}
		table={caslib=&caslib_ref, name=&in_mem_tbl}
		casOut={caslib=&caslib_ref, name='casgradboost_scored', replace=True}
		copyVars={&target}
		encodeName=TRUE
		assessOneRow=TRUE
	;
run;


/* assess model */

proc cas;
	percentile.assess /
		table={caslib=&caslib_ref, name="casgradboost_scored"}
		event="1"
		response=&target
		inputs=&predictedtarget
		cutStep=0.001
		casOut={caslib=&caslib_ref, name='casgradboost_assess', replace=True}
	;
run;
title;

proc sgplot data=public.casgradboost_assess_roc;
    series x=_FPR_ y=_Sensitivity_ / group=_Event_;
	title "ROC Curve";
run;
title;

data ks_value (where=(_KS_=1));
	set public.casgradboost_assess_roc;
run;

*KS2_ is the highest KS given cutSteps;
proc print data=ks_value;
run;


/* model interpretability - partial dependency */

proc cas;
	explainModel.partialDependence /
		table={caslib=&caslib_ref, name=&in_mem_tbl}
		seed=12345
		modelTable={caslib=&caslib_ref, name="casgradboost_astore"}
		predictedTarget=&predictedtarget
		analysisVariable=&partial_dep_var
		inputs=&inputs
		output={casOut={caslib=&caslib_ref, name='partial_dependency', replace=True}}
		;
run;


/* model interpretability - single obs */

data public.shapley_obs;
	set &libname_in_mem_tbl;
	if account_id = 5;
run;

data public.shapley_train;
	set &libname_in_mem_tbl;
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

%let sample_size=500;

data public.shapley_train;
	set &libname_in_mem_tbl;
run;

proc cas;
	sampling.srs /
		table={caslib=&caslib_ref, name=&in_mem_tbl}
		fixedObs=&sample_size
		seed=12345
		output={casOut={caslib=&caslib_ref, name="shapley_sample", replace=true},
				copyvars="ALL"};
run;

%macro shapley;

%do i=1 %to &sample_size.;
	data public.shapley_temp;
		set public.shapley_sample (firstobs=&i. obs=&i.);
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
		data public.shapley_rows;
			set public.shapley_transpose;
		run;
		data public.shapley_cols;
			set public.shapleyvalues;
		run;
		%end;
	%else %do;
		data public.shapley_rows(append=force);
			set public.shapley_transpose;
		run;
		data public.shapley_cols(append=force);
			set public.shapleyvalues;
		run;
		%end;
%end;

%mend shapley;
%shapley;

* may need to unload global scope tables first and save as perm tables;
proc casutil;
	promote casdata='shapley_rows' incaslib=&caslib_ref 
			casout='shapley_rows' outcaslib=&caslib_ref;
run;
proc casutil;
	promote casdata='shapley_cols' incaslib=&caslib_ref 
			casout='shapley_cols' outcaslib=&caslib_ref;
run;


/* charting HyperSHAP values */

data public.shapley_avg;
	set public.shapley_cols;
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

proc sgplot data=public.shapley_cols;
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
			I_indicator = round(&predicted,1);
			diff = abs(&target_noquote - I_indicator);
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

/*proc surveyselect data=&libname_in_mem_tbl
	seed=12345
	method=balbootstrap
	reps=10
	outhits
	out=public.resample;
	samplingunit &target_noquote;
	strata occupation_risk;
run;*/

*proc surveyselect data=Customers method=seq n=(8 12 20 10)
                  reps=4 seed=40070 ranuni out=SampleRep;


/* bias assessment */

proc astore;
	describe rstore=public.casgradboost_astore;
run;

* run separately for each var;
* results have to be copied since output to table is not available;
proc cas;
	fairAITools.assessBias /
		table = {caslib=&caslib_ref, name=&in_mem_tbl},
		modelTable = "casgradboost_astore",
		modelTableType = "astore",
		response = &target,
		predictedVariables = {&predictedvars},
		responseLevels={"0", "1"},
		sensitiveVariable = &bias_var;
run;
