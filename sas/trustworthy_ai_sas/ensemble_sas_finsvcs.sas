cas casauto sessopts=(caslib=casuser, metrics=true, timeout=900);
libname sess_lib cas caslib=casuser;

/* set macro variables */

%let in_mem_tbl = 'financial_services_prep';
%let caslib_ref = 'public';
%let libname_in_mem_tbl = public.financial_services_prep;
%let libname_shapcols = sess_lib.shapley_cols;
%let libname_shaprows = sess_lib.shapley_rows;
%let libname_shaptemp = sess_lib.shapley_temp;
%let libname_shaptrain = sess_lib.shapley_train;
%let libname_shapobs = sess_lib.shapley_obs;
%let libname_shapsample = sess_lib.shapley_sample;
%let libname_shaptp = sess_lib.shapley_transpose;
%let libname_shapvals = sess_lib.shapleyvalues;
%let libname_shapavg = sess_lib.shapley_avg;
%let promote_shaprows = 'ensemble_sas_finsvcs_shaprows';
%let promote_shapcols = 'ensemble_sas_finsvcs_shapcols';
%let predictedtarget = 'P_event_indicator1';
%let excluded_cols = event_indicator analytic_partition account_id
						id_important_activity id_direct_contact 
						id_current_fs_relationship;
%let epcode_dir = "/.../dmcas_epscorecode_ensemble_0807.sas";
/* The astores necessary to run scorecode should be listed at the top 
	of the scorecode downloaded directly from Model Studio.
	The astores will need to be copied and pasted into code below. */

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

/* This is where the Model Studio models sit */
proc casutil incaslib=models;
	list files;
run;

/* This is where the Model Manager registered models sit */
proc casutil incaslib=modelstore;
	list files;
run;

/* Action to describe a specific astore */
proc cas;
	aStore.describe result=R /
      rstore={caslib='ModelStore', name='_R5O4UW0M5KCP16S6F5AQM56NF.sashdat'}
		epcode = TRUE;
		describe R;
		ds2_code = R['epcode'];
		saveresult R["InputVariables"] casout='sas_model_inputs' replace;
		print(R['Key']);
		print(R['Description']);
		print(ds2_code);
run;

/* model interpretability - single obs */

data &libname_shapobs; *(drop=&excluded_cols);
	set &libname_in_mem_tbl;
	if account_id = 6224;
run;

data &libname_shaptrain; *(drop=&excluded_cols);
	set &libname_in_mem_tbl;
run;

proc cas;
	source epcode;
        %include &epcode_dir;
    endsource;
	table.columnInfo result=C /
		table={name='shapley_obs'};
	explainModel.shapleyExplainer / 
		table={name="shapley_train"}
		query={name="shapley_obs"}
		modelTables={{caslib='models', name='_5YX6KS4UJT0O9HDIO303UKVXY_ast.sashdat'},
					{caslib='models', name='_9TWDG989Y32AT3PJW0V62CBKX_ast.sashdat'}
					{caslib='models', name='_6U59G8CN3LICIM5MF4AFSSUZM_ast.sashdat'}}
		modelTableType="astore"
		code = epcode
		predictedTarget=&predictedtarget
		inputs=C.ColumnInfo[,'Column']
		depth=1
		outputTables={includeAll=TRUE};
run;

/* model interpretability - multiple obs */

%let sample_size=2;

data &libname_shaptrain;
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
	data &libname_shaptemp;
		set &libname_shapsample (firstobs=&i. obs=&i.);
	run;
	proc cas;
		source epcode;
        	%include &epcode_dir;
    	endsource;
		table.columnInfo result=C /
			table={name='shapley_obs'};
		explainModel.shapleyExplainer / 
			table={name="shapley_train"}
			query={name="shapley_temp"}
			modelTables={{caslib='models', name='_5YX6KS4UJT0O9HDIO303UKVXY_ast.sashdat'},
						{caslib='models', name='_9TWDG989Y32AT3PJW0V62CBKX_ast.sashdat'}
						{caslib='models', name='_6U59G8CN3LICIM5MF4AFSSUZM_ast.sashdat'}}
			modelTableType="astore"
			code = epcode
			predictedTarget=&predictedtarget
			inputs=C.ColumnInfo[,'Column']
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
		data &libname_shaprows;
			set &libname_shaptp;
		run;
		data &libname_shapcols;
			set &libname_shapvals;
		run;
		%end;
	%else %do;
		data &libname_shaprows(append=force);
			set &libname_shaptp;
		run;
		data &libname_shapcols(append=force);
			set &libname_shapvals;
		run;
		%end;
%end;

%mend shapley;
%shapley;

/* charting HyperSHAP values */

data &libname_shapavg;
	set &libname_shapcols;
	ShapleyValue = abs(ShapleyValue);
run;

title;
proc sgplot data=&libname_shapavg;
	title "Average Absolute HyperSHAP Values";
  	hbar Variable /
    response=ShapleyValue
    stat=mean
	categoryorder=respdesc;
run;
title;

proc sgplot data=&libname_shapcols;
	title "HypeSHAP Values by Attribute";
	heatmap x=ShapleyValue y=Variable /
		colormodel=(blue yellow red)
		nxbins=25
   		;
run;

* may need to unload global scope tables first and/or save as perm tables;

proc casutil;
	promote casdata='shapley_rows'
			casout=&promote_shaprows outcaslib=&caslib_ref;
run;

proc casutil;
	promote casdata='shapley_cols'
			casout=&promote_shapcols outcaslib=&caslib_ref;
run;
