/* Session-Scope caslib
When a caslib is defined without including the GLOBAL option, 
the caslib is a session-scoped caslib. When a table is loaded to the CAS 
server with session-scoped caslib, the table is available to that specific 
CAS user session only. */

/* Global-Scope caslib
When a caslib is defined using a CASLIB statement with the GLOBAL option, 
the caslib is defined as a global-scoped caslib. The global-scoped caslib 
and associated table could be made available to other users in the CAS 
server by updating the caslib’s access controls. */

/* Default caslib
The default caslib is CASUSER(userid), which is a personal caslib, and it 
is the "active" caslib for the session as defined (e.g., CASAUTO). 
The "active" caslib can be changed by altering the caslib= statement in 
the session definition. */

/* CAS Libname
Associates a SAS libref with tables on the SAS Cloud Analytic Services 
server. The default libref is typically WORK or SASUSER. Creating a libname
allows the user to view tables generated as part of the code the under 
"Libraries" tab and to access those tables in SAS data steps and procs. */

cas casauto sessopts=(caslib=casuser, metrics=true, timeout=900);
libname chris cas caslib=casuser;


/* set macro variables */

%let in_mem_tbl = 'financial_services_prep';
%let caslib_ref = 'public';
%let libname_in_mem_tbl = public.financial_services_prep;
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
			"age", "amount", "at_current_job_1_year", 
			"credit_history_mos", "credit_score", "debt_to_income",
			"gender", "job_in_education", "job_in_hospitality",
			"net_worth", "num_dependents", "num_transactions"};
%let inputs_noquoteorcomma = 
			age amount at_current_job_1_year
			credit_history_mos credit_score debt_to_income
			gender job_in_education job_in_hospitality
			net_worth num_dependents num_transactions;
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
		casOut={name="gbtree_sas_model", replace=True}
		saveState={name="gbtree_sas_astore", replace=True}
    	;
run;

proc cas;
	decisionTree.gbtreeCode /
		modelTable={name="gbtree_sas_model"}
		code={casOut={name='gbtree_sas_scorecode', 
						replace=True, promote=False}}
	;
run;

proc cas;
	decisionTree.gbtreeScore /
		modelTable={name="gbtree_sas_model"}
		table={caslib=&caslib_ref, name=&in_mem_tbl}
		casOut={name='gbtree_sas_score', replace=True}
		copyVars={&target}
		encodeName=TRUE
		assessOneRow=TRUE
	;
run;


/* assess model */

proc cas;
	percentile.assess /
		table={name="gbtree_sas_score"}
		event="1"
		response=&target
		inputs=&predictedtarget
		cutStep=0.001
		casOut={name='gbtree_sas_assess', replace=True}
	;
run;

proc sgplot data=chris.gbtree_sas_assess_roc;
    series x=_FPR_ y=_Sensitivity_ / group=_Event_;
	title "ROC Curve";
run;
title;

data ks_value (where=(_KS_=1));
	set chris.gbtree_sas_assess_roc;
run;

*KS2_ is the highest KS given cutSteps;
proc print data=ks_value;
run;


/* model interpretability - partial dependency */

proc cas;
	explainModel.partialDependence /
		table={caslib=&caslib_ref, name=&in_mem_tbl}
		seed=12345
		modelTable={name="gbtree_sas_astore"}
		predictedTarget=&predictedtarget
		analysisVariable=&partial_dep_var
		inputs=&inputs
		output={casOut={name='partial_dependency', replace=True}}
		;
run;


/* model interpretability - single obs */

data chris.shapley_obs;
	set &libname_in_mem_tbl;
	if account_id = 5;
run;

data chris.shapley_train;
	set &libname_in_mem_tbl;
run;

proc cas;
   explainModel.shapleyExplainer / 
		table={name="shapley_train"}
		query={name="shapley_obs"}
		modelTable={name="gbtree_sas_astore"}
		modelTableType="astore"
		predictedTarget=&predictedtarget
		inputs=&inputs
		depth=1
		outputTables={includeAll=TRUE};
run;

/* model interpretability - multiple obs */

%let sample_size=500;

data chris.shapley_train;
	set &libname_in_mem_tbl;
run;

proc cas;
	sampling.srs /
		table={caslib=&caslib_ref, name=&in_mem_tbl}
		fixedObs=&sample_size
		seed=12345
		output={casOut={name="shapley_sample", replace=true},
				copyvars="ALL"};
run;

%macro shapley;

%do i=1 %to &sample_size.;
	data chris.shapley_temp;
		set chris.shapley_sample (firstobs=&i. obs=&i.);
	run;
	proc cas;
   		explainModel.shapleyExplainer / 
			table={name="shapley_train"}
			query={name="shapley_temp"}
			modelTable={name="gbtree_sas_astore"}
			modelTableType="astore"
			predictedTarget=&predictedtarget
			inputs=&inputs
			depth=1
			outputTables={includeAll=TRUE}
			;
		transpose.transpose /
			table={name="shapleyvalues"}
   			id={"variable"}
   			casOut={name="shapley_transpose", replace=true}
			;
   	run;
	quit;
	%if &i. = 1 %then %do;
		data chris.shapley_rows;
			set chris.shapley_transpose;
		run;
		data chris.shapley_cols;
			set chris.shapleyvalues;
		run;
		%end;
	%else %do;
		data chris.shapley_rows(append=force);
			set chris.shapley_transpose;
		run;
		data chris.shapley_cols(append=force);
			set chris.shapleyvalues;
		run;
		%end;
%end;

%mend shapley;
%shapley;

* may need to unload global scope tables first and save as perm tables;
proc casutil;
	promote casdata='shapley_rows' incaslib=chris 
			casout='gbtree_action_sas_finsvcs_shapley_rows' outcaslib=&caslib_ref;
run;
proc casutil;
	promote casdata='shapley_cols' incaslib=chris 
			casout='gbtree_action_sas_finsvcs_shapley_cols' outcaslib=&caslib_ref;
run;

/* charting HyperSHAP values */

data chris.shapley_avg;
	set chris.shapley_cols;
	ShapleyValue = abs(ShapleyValue);
run;

proc sgplot data=chris.shapley_avg;
	title "Average Absolute HyperSHAP Values";
  	hbar Variable /
    response=ShapleyValue
    stat=mean
	categoryorder=respdesc;
run;
title;

proc sgplot data=chris.shapley_cols;
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
		output={casOut={name="gbtree_sas_kfold", replace=true},
				copyVars='ALL',
				foldname='_fold_'}
	;
run;

%macro kfold;

%do i=1 %to 10;

		proc cas;
    	decisionTree.gbtreeTrain /
			table={name='gbtree_sas_kfold', where="_fold_ = " || &i.}
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
			casOut={name="gbtree_sas_temp", replace=True}
    		;

		decisionTree.gbtreeScore /
			modelTable={name="gbtree_sas_temp"}
			table={name='gbtree_sas_kfold', where="_fold_ = " || &i.}
			casOut={name='gbtree_sas_temp_scored', replace=True}
			copyVars={&target}
			encodeName=TRUE
			assessOneRow=TRUE
			;
  		run;

		data chris.temp;
			set chris.gbtree_sas_temp_scored;
			I_indicator = round(&predicted,1);
			diff = abs(&target_noquote - I_indicator);
		run;

		proc cas;

		freqTab.freqTab /
      		table={name='temp'},
			tabulate={'I_indicator'},
			outputTables={includeAll=TRUE}
		;
		run;

		data chris.onewayfreqs;
			set chris.onewayfreqs;
			_fold_ = &i.;
		run;

		%if &i. = 1 %then %do;
			data chris.kfold_output;
				set chris.onewayfreqs;
			run;
		%end;
		%else %do;
			data chris.kfold_output(append=force);
				set chris.onewayfreqs;
			run;
		%end;

%end;

%mend kfold;
%kfold;

proc casutil;
	promote casdata='kfold_output' incaslib=chris 
			casout='gbtree_action_sas_finsvcs_kfold_output' outcaslib=&caslib_ref;
run;

/* bias assessment */

proc astore;
	describe rstore=chris.gbtree_sas_astore;
run;

* run separately for each var;
* results have to be copied since output to table is not available;
proc cas;
	fairAITools.assessBias /
		table = {caslib=&caslib_ref, name=&in_mem_tbl},
		modelTable = "gbtree_sas_astore",
		modelTableType = "astore",
		response = &target,
		predictedVariables = {&predictedvars},
		responseLevels={"0", "1"},
		sensitiveVariable = &bias_var;
run;
