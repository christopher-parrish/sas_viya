cas casauto sessopts=(caslib=casuser, metrics=true, timeout=900);
libname cp cas caslib=casuser;
caslib _all_ assign;

/* 
Blog: 9 for SAS9 – Top Tips for SAS 9 Programmers Moving to SAS Viya
https://blogs.sas.com/content/sgf/2024/01/30/9-for-sas9-top-tips-for-sas-9-programmers-moving-to-sas-viya/
*/

/******************************
* Load CAS Tables into Memory *
*******************************/ 

%let caslib_ref = 'casuser';

proc casutil;
	list files incaslib=&caslib_ref;
	list tables incaslib=&caslib_ref;
run;

%let tbl_load = 'AML_BANK_PREP.sashdat'; *case sensitive*;
%let tbl_out = 'aml_bank_prep';

proc casutil;
	load
	casdata=&tbl_load incaslib=&caslib_ref
	casout=&tbl_out outcaslib=&caslib_ref
	promote;
run;

/****************************
* Load SAS Dataset into CAS *
*****************************/

data casuser.table;
	set work.table; *specify directory*;
run;