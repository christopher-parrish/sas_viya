data casuser.myair;
   set sashelp.air;
run;

/* Uses CAS with massive parallel processing and in-memory tables */
proc carima data=casuser.myair outest=casuser.cariest outfor=casuser.carifor;
  id date interval=month;
  identify air;
  estimate q=(1)(12) noint  method = ML transform = log diff = (1,12);
  forecast lead = 4;
run;

/* Uses SAS Compute (SAS 9 runtime) */
proc arima data=casuser.myair;
   identify var=air(1,12);
   estimate q=(1)(12) noint method=ml;
   forecast id=date interval=month printall out=b;
run;