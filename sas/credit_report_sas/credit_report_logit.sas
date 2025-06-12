proc logselect data=casuser.credit_report_woe;
    model customer_event (event = '0') = 
            woe_num_credit_accounts_open
            woe_age
            woe_debt_to_income
            woe_credit_utilization_ratio
            woe_length_of_last_job_mos
            woe_credit_history_mos
            woe_credit_score
            woe_scheduled_payments_per_month;
    selection method=NONE;
    output out=casuser.logselect_out allstat copyvars=account_id;
run;