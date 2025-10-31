%let server = '[account].snowflakecomputing.com';
%let warehouse = '*';
%let warehouse_nq = *;
%let database = '*';
%let database_nq = *;
%let schema = '*';
%let schema_nq = *;
%let user = '*';
%let user_nq = *;
%let password = '*';
%let password_nq = *;

proc sql noerrorstop;
 connect to snow as conn(server=&server user=&user password=&password 
						warehouse=&warehouse database=&database schema=&schema);

 select * from connection to conn(
	SELECT count(*) as row_count FROM aml_bank);

 disconnect from conn;
 quit;

libname snowlib snow server=&server. user=&user_nq. password=&password_nq. 
						warehouse=&warehouse_nq. database=&database_nq. schema=&schema_nq.;

proc print data=snowlib.cars;
	where make='acura';
run;

proc sql noerrorstop;
 connect to snow as conn(server=&server user=&user password=&password 
						warehouse=&warehouse database=&database schema=&schema);
 create table casuser.test as
 select * from connection to conn(
	select * from aml_bank);

 disconnect from conn;
 quit;