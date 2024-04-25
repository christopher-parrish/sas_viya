%let charlist = customer_event_char gender_char currently_employed_char homeowner_char dpd_30_currently_char responded_to_campaign_12_mo_char dpd_60_over_past_12_mos_char bankruptcy_indicator_char
                unpaid_support_indicator_char collection_agency_indicator_char marital_status_char last_campaign_account_type_char id_important_activity_char id_direct_contact_char 
                id_current_relationship_char campaign_peer_id_char;

data credit_report;
	set casuser.credit_report;
run;

title 'Heat Map of Response to Campaigns by Peer ID and Account Type';
title2 'Higher Frequencies Indicate Customers Responded in Higher Numbers to Targeted Campaigns by Peer (Bank) Group';
proc sgplot data=credit_report;
    heatmap x=campaign_peer_id_char y= last_campaign_account_type_char;
run;
title '';

title "Campaign Performance";
proc sgpanel data=credit_report;
    panelby responded_to_campaign_12_mo_char customer_event_char / novarname ;
    hbar campaign_peer_id_char;
run;
title '';

%let input_columns =
			(account_id performance_period_stress customer_event gender currently_employed homeowner dpd_30_currently responded_to_campaign_12_mos dpd_60_over_past_12_mos 
            bankruptcy_indicator unpaid_support_indicator collection_agency_indicator marital_status last_campaign_account_type	num_soft_inquiries_3_mos num_hard_inquiries_3_mos
            num_addresses_reported num_credit_accounts_open num_credit_accounts_closed age debt_to_income credit_utilization_ratio length_of_last_job_mos credit_history_mos 
            credit_score scheduled_payments_per_month id_important_activity id_direct_contact id_current_relationship campaign_peer_id geographical_code ssn customer_event_char
            gender_char currently_employed_char homeowner_char dpd_30_currently_char responded_to_campaign_12_mo_char dpd_60_over_past_12_mos_char bankruptcy_indicator_char
            unpaid_support_indicator_char collection_agency_indicator_char marital_status_char last_campaign_account_type_char id_important_activity_char id_direct_contact_char
            id_current_relationship_char campaign_peer_id_char);

proc binning data=casuser.credit_report numbin=8 woe;
   input num_credit_accounts_open
         age
         debt_to_income
         credit_utilization_ratio
         length_of_last_job_mos
         credit_history_mos
         credit_score
         scheduled_payments_per_month;
   target customer_event/event='1';
   output out=casuser.credit_report_woe copyvars=&input_columns;
run;

data credit_report_woe;
    set casuser.credit_report_woe;
    rename BIN_num_credit_accounts_open=woe_num_credit_accounts_open
            BIN_age=woe_age
            BIN_debt_to_income=woe_debt_to_income
            BIN_credit_utilization_ratio=woe_credit_utilization_ratio
            BIN_length_of_last_job_mos=woe_length_of_last_job_mos
            BIN_credit_history_mos=woe_credit_history_mos
            BIN_credit_score=woe_credit_score
            BIN_scheduled_payments_per_month=woe_scheduled_payments_per_month;
run;

proc logselect data=credit_report_woe;
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
    output out=logselect_out allstat copyvars=account_id;
run;

%let target_score = 600;
%let target_odds = 30;
%let points_to_double_the_odds = 20;
%let factor = &points_to_double_the_odds / log(2);
%let offset = &target_score - &factor * log(&target_odds);

data credit_report_score;
    set logselect_out;
    log_odds = _XBETA_;
    prob_good_credit = _PRED_;
    score = &offset + (&factor * log_odds);
run;

proc sgplot data=credit_report_score;
    histogram score;
    title 'Distribution of Scores';
run;
title '';

proc print data=credit_report_score (obs=25);
    var account_id log_odds prob_good_credit score;
run;
