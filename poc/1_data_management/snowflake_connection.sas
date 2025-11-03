%let server = 'xxx.snowflakecomputing.com';
%let warehouse = 'xxx';
%let warehouse_nq = xxx;
%let database = 'xxx';
%let database_nq = xxx;
%let schema = 'xxx';
%let schema_nq = xxx;
%let user = 'xxx';
%let user_nq = xxx;
%let password = 'xxx';
%let password_nq = xxx;

libname snowlib snow server=&server. user=&user_nq. password=&password_nq. 
						warehouse=&warehouse_nq. database=&database_nq. schema=&schema_nq.;

proc print data=snowlib.AML_BANK;
	where account_id=1;
run;


proc sql noerrorstop;
 connect to snow as conn(server=&server user=&user_nq password=&password 
						warehouse=&warehouse_nq database=&database_nq schema=&schema_nq);

 select * from connection to conn(
	SELECT count(*) as row_count FROM AML_BANK);

 disconnect from conn;
 quit;



proc sql noerrorstop;
 connect to snow as conn(server=&server user=&user_nq password=&password 
						warehouse=&warehouse_nq database=&database_nq schema=&schema_nq);
 create table aml_bank_table as
 select * from connection to conn(
	SELECT * FROM AML_BANK WHERE account_id < 6);

 disconnect from conn;
 quit;