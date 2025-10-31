/*################
  ### SAS CODE ###
  ################*/

  cas casauto sessopts=(caslib=casuser, metrics=true, timeout=900);
  libname cp cas caslib=casuser;
  caslib _all_ assign;

/*###########################
  ### Set Macro Variables ###
  ###########################*/
  
%let in_mem_tbl = 'financial_services_prep';
%let caslib_ref = 'casuser';
%let model_name = 'gbtree_sas_finsvcs';
%let libname_in_mem_tbl = casuser.financial_services_prep;
%let target = 'event_indicator';
%let predictedtarget = 'P_event_indicator1';
%let excluded_cols = {"event_indicator", "analytic_partition", "account_id",
		"id_important_activity", "id_direct_contact", "id_current_fs_relationship"};
%let astore_tbl = &model_name || '_astore';
%let model_tbl = &model_name || '_model';
%let score_tbl = &model_name || '_score';
%let scorecode_tbl = &model_name || '_scorecode';
%let assess_tbl = &model_name || '_assess';
%let assess_roc_tbl = &model_name || '_assess_roc';

/*#############################
  ### Identify Table in CAS ###
  #############################*/

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
  quit;

/*########################
  ### Model Parameters ###
  ########################*/

  proc cas;
	table.columnInfo result=C /
		table={caslib=&caslib_ref, name=&in_mem_tbl};
		inputs = C.ColumnInfo[,'Column'];
		inputs = inputs - &excluded_cols;
 
/*###################
  ### Train Model ###
  ###################*/

	decisionTree.gbtreeTrain /
		table={caslib=&caslib_ref, name=&in_mem_tbl}
		target=&target
		inputs=inputs
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
		casOut={name=&model_tbl, replace=True}
		saveState={name=&astore_tbl, replace=True}
    	;
run;
quit;

/*########################
  ### Write Score Code ###
  ########################*/

proc cas;
	decisionTree.gbtreeCode /
		modelTable={name=&model_tbl}
		code={casOut={name=&scorecode_tbl, replace=True, promote=False}}
	;
run;
quit;

/*########################
  ###   Score Model    ###
  ########################*/

proc cas;
	decisionTree.gbtreeScore /
		modelTable={name=&model_tbl}
		table={caslib=&caslib_ref, name=&in_mem_tbl}
		casOut={name=&score_tbl, replace=True}
		copyVars={&target}
		encodeName=TRUE
		assessOneRow=TRUE
	;
run;
quit;

/*########################
  ###   Assess Model   ###
  ########################*/

proc cas;
	percentile.assess /
		table={name=&score_tbl}
		event="1"
		response=&target
		inputs=&predictedtarget
		cutStep=0.001
		casOut={name=&assess_tbl, replace=True}
	;
run;
quit;

proc cas;
	dataStep.runCode /
		code = "data test;
				set public.casgradboost_assess_roc;
				if _KS_ = 1;
				run;";
run;
quit;

/* _KS2_ is the highest KS given cutSteps */
proc print data=cp.test;
run;

proc cas;
		table.tableExists result=code /
			caslib=&caslib_ref, name='casgradboost_assess_roc';
			if code['exists'] = 0 then do;
			print "The CAS table does not exist";
			end;

			if code['exists'] = 1 then do;
			print "The CAS table has a session scope";
			end;

			if code['exists'] = 2 then do;
			print "The CAS table has a global scope";
			end;
run;
quit;