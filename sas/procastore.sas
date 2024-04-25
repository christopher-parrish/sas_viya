cas casauto sessopts=(caslib=casuser, metrics=true, timeout=900);

proc astore;
  upload rstore=public.financial_services_astore
         store="/greenmonthly-export/ssemonthly/homes/Chris.Parrish@sas.com/financial_services.astore";
run;

proc casutil;
	promote casdata='financial_services_astore' incaslib="public" 
			casout='financial_services_astore' outcaslib="public";
run;
quit;