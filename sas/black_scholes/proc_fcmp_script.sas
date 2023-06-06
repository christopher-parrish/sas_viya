proc fcmp outlib = work.functions.bs;
	function black_scholes(notional, vol, strike_price, spot_price, time_to_mat, risk_free_rate);
    	d1 = (log(spot_price/strike_price) + (risk_free_rate+vol**2/2)*time_to_mat)/(vol*sqrt(time_to_mat));
    	d2 = d1 - vol*sqrt(time_to_mat);
    	option_price = cdf('NORMAL', d1)*spot_price-cdf('NORMAL', d2)*strike_price*exp(-risk_free_rate*time_to_mat);
    	option_val = notional * option_price;
    	return (option_val);
	endfunc;
run;

options cmplib=work.functions;

data _null_;
	notional = 200;
	vol = 0.5;
	strike_price = 15;
	spot_price = 5;
	time_to_mat = 1;
	risk_free_rate = 1;
	option_val = black_scholes(notional, vol, strike_price, spot_price, time_to_mat, risk_free_rate);
	put option_val =;
run;

/* prints 'put' statement in the log */
