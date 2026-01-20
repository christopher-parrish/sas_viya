data test; /* Create a new dataset named 'test' */
    set sashelp.cars; /* Read all observations from the 'sashelp.cars' dataset */
run; /* End of DATA step */

proc contents data=test;
run;

/* create a model from the test dataset that predicts mpg */
data test_for_model;
    set work.test;
    if mpg_highway ne .;
run;
proc gradboost data=test_for_model outmodel=work.gbmodel_mpg;
    target mpg_highway / level=interval;
    input MSRP Invoice EngineSize Cylinders Horsepower Weight Wheelbase Length / level=interval;
    input Make Type Origin DriveTrain / level=nominal;
run;