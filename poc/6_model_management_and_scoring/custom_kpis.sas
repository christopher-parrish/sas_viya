cas _mmcas_;
caslib _all_ assign;

%let _MM_PerfExecutor = 1;
%let _MM_ProjectUUID = %nrstr(2c7e3cf0-65f5-4daf-b8f3-246091d0372e);
%let _MM_TargetVar = ml_indicator;
%let _MM_TargetLevel = BINARY;
%let _MM_PredictedVar = ;
%let _MM_TargetEvent = 1;
%let _MM_EventProbVar = P_ml_indicator1;
%let _MM_KeepVars = P_ml_indicator1;
%let _MM_CAKeepVars = REASON VALUE YOJ CLAGE CLNO DEBTINC DELINQ DEROG 
                      JOB LOAN MORTDUE NINQ;
%let _MM_Trace = OFF;
%let _MM_Max_Bins = 10;
%let _MM_PerfOutCaslib = ModelPerformanceData;
%let _MM_PerfInCaslib = casuser;
%let _MM_Perf_InTablePrefix = aml_bank_prep;
%let _MM_PerfStaticTable = ;
%let _MM_ForceRunAllData = N;
%let _MM_RunScore = Y;
%let _MM_SAVEPERFRESULT = Y;
%let _MM_JobID = %nrstr(6e80398b-556c-4f9d-8373-80a45161a570);
%let _MM_ModelID = %nrstr(75c21f18-873c-4625-9d06-587dcc0eb559);
%let _MM_ModelName = %nrstr(logit_python_api_aml_bank_workbench);
%let _MM_ModelFlag = 0;
%let _MM_ScoreCodeType = DS2EP;
%let _MM_ScoreCodeURI = /files/files/b7da0a92-b076-45cd-aa85-485aa03f5b67;
%let _MM_ScoreAstURI = ;
%let _MM_aStoreLocation=ModelStore._93034785EFAC7B4BFF6BA4E4E_AST ;

%mm_performance_monitor
(
    perfLib=&_MM_PerfInCaslib,
    perfDataNamePrefix=&_MM_Perf_InTablePrefix,
    mm_mart=&_MM_PerfOutCaslib,
    runScore=&_MM_RunScore,
    scorecodeURI=&_MM_ScoreCodeURI
);

%put &syserr;
%put &syscc;

/* View the performance monitoring results. */
libname mm_mart cas caslib="&_MM_PerfOutCaslib" tag="&_MM_ProjectUUID";

/* View a list of the MM_MART library tables. */
proc datasets lib=mm_mart;
run;