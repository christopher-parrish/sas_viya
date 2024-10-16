cas casauto sessopts=(caslib=casuser, metrics=true, timeout=900);

/* example where .astore was manually uploaded to SAS Viya */

proc astore;
  upload rstore=public.financial_services_astore
         store="/path-on-sas-viya-compute-server/financial_services.astore";
run;

proc casutil;
	promote casdata='financial_services_astore' incaslib="public" 
			casout='financial_services_astore' outcaslib="public";
run;
quit;