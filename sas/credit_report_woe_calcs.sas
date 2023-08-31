cas casauto sessopts=(caslib=casuser);
%let in_mem_tbl = 'credit_report';
%let caslib_ref = 'casuser';
%let cas_out = 'credit_report_woe';

proc cas;
table.columnInfo result=C /
		table={caslib=&caslib_ref, name=&in_mem_tbl};
all_columns = C.ColumnInfo[,'Column'];
input_columns={"num_credit_accounts_open",
				"num_credit_accounts_closed",
				"age",
				"debt_to_income",
				"credit_utilization_ratio",
				"length_of_last_job_mos",
				"credit_history_mos",
				"credit_score",
				"scheduled_payments_per_month"
				};
target = {"customer_event"};
copyvars_columns=all_columns-input_columns;

dataPreprocess.transform status=rc /
    table={name=&in_mem_tbl}
    casout={name=&cas_out, promote=0, replace=1}
    requestPackages={{discretize={method="WOE", arguments={minNBins=3 maxNBins=7}}, 
    					inputs=input_columns, targets=target, events='1', output={scoreWOE=TRUE}}}
    outVarsNameGlobalPrefix="woe"
    copyVars=copyvars_columns;

if rc.statusCode != 0 then do;
    print "Error running action in CASL";
    exit 3;
end;
run;

* may need to unload global scope tables first and save as perm tables;
proc casutil;
	promote casdata=&cas_out incaslib=&caslib_ref 
			casout=&cas_out outcaslib=&caslib_ref;
run;

quit;