proc logselect data=casuser.aml_bank_prep;
    model ml_indicator (event = '0') = 
        marital_status_single
        checking_only_indicator
        prior_ctr_indicator
        address_change_2x_indicator
        cross_border_trx_indicator
        in_person_contact_indicator
        linkedin_indicator
        citizenship_country_risk
        distance_to_employer
        distance_to_bank;
    selection method=NONE;
    code file="/innovationlab-export/innovationlab/homes/Chris.Parrish@sas.com/chris_git/sas/logit_sas/logit_ds1code_score";
    output out=casuser.logselect_out allstat copyvars=account_id;
run;