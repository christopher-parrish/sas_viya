/* 
Summary:
1) Invest funds among available assets
2) Minimum 10% investment per asset
3) Maximize expected return
4) Limit total pairwise covariance
5) Invest 100% of allocated funds

Solution Approaches:

Conventional
	One binary variable per asset (in or out of the portfolio)
	One continuous variable per asset (level of investment)
	Nonlinear covariance constraint
	Mixed integer nonlinear program (MINLP)
Alternative
	Randomly select assets in each of multiple trials
	Eliminate binary variables
	Solve one nonlinear program (NLP) per trial, in parallel

Example script found in video on: https://www.sas.com/en_us/software/optimization.html
*/

/* ASSETS table as CAS table or SAS dataset */

libname mycaslib cas caslib=casuser;

proc optmodel;
	/* index set of assets with historical mean return and covariance values -
		11 assets in total */
	set ASSETS;
	num return (ASSETS);
	num cov {ASSETS,ASSETS} init 0;
	read data mycaslib.pf_means into ASSETS=[_n_] return;
	read data mycaslib.pf_cov into [asset1 asset2] cov cov[asset2,asset1]=cov;
	num riskLimit init 0.0025;
	num minThreshold init 0.1;
	num numTrials = 20;

	/* declare NLP problem for fixed set of assets with decision variables 
		with objective of maximizing returns and constraints to limit risk and invest all cash */
	set ASSETS_THIS;
	var AssetPropVar {ASSETS_THIS} >= minThreshold <=1;
	max Expected Return = 
			sum{i in ASSETS_THIS} return [i] * AssetPropVar[i];
	con RiskBound:
			sum{i in ASSETS_THIS, j in ASSETS_THIS} cov[i,j] *
			AssetPropVar[i] * AssetPropVar[j] <= riskLimit;
	con TotalPortfolio:
			sum{asset in ASSETS_THIS} AssetPropVar[asset] = 1;

	num infinity = constant('BIG');
	num best_objective init-infinity;
	set INCUMBENT;
	num sol {INCUMBENT};

	num overall_start;
	overall_start = time();
	set TRIALS = 1..numTrials;
	num start (TRIALS);
	num finish {TRIALS};
	call streaminit(1);
	/* concurrent for loop for trials in parallel in SAS Viya */
	cofor {trial in TRIALS} do;
		start[trial] = time() - overall_start;
		put;
		put trial=;
		ASSETS_THIS = {i in ASSETS: rand('UNIFORM') < 0.5};
		put ASSETS_THIS=;
		solve with nlp / logfreq=0;
		put ASSETS_THIS=;
		put _solution_status_=;
		if _solution_status_ in {'OPTIMAL', 'BEST_FEASIBLE'} then do;
			put ExpectedReturn = ASSETS_THIS=;
			if best_objective < ExpectedReturn then do;
				best_objective = ExpectedReturn;
				INCUMBENT = ASSETS_THIS;
				put best_objective = INCUMBENT=;
				put RiskBound.body = RiskBound.ub=;
				put TotalPortfolio.body = TotalPortfolio.ub=;
				for {i in INCUMBENT} sol[i] = AssetPropVar[i];
			end;
		end;
		finish[trial] = time() - overall_start;
	end;
	put best_objective = INCUMBENT=;
	for {i in INCUMBENT} put i sol[i];
	create data solution from [Asset]=INCUMBENT Investment=sol;
quit;
   print x;
 quit;