*****************************************;
** SAS Scoring Code for PROC Logistic;
*****************************************;

length I_advisor_event_indicator $ 12;
label I_advisor_event_indicator = 'Into: advisor_event_indicator' ;
label U_advisor_event_indicator = 
'Unnormalized Into: advisor_event_indicator' ;
format U_advisor_event_indicator BEST12.0;

label P_advisor_event_indicator0 = 'Predicted: advisor_event_indicator=0' ;
label P_advisor_event_indicator1 = 'Predicted: advisor_event_indicator=1' ;

drop _LMR_BAD;
_LMR_BAD=0;

*** Check interval variables for missing values;
if nmiss(sf_face_2_face,sf_call_outbound,sf_call_inbound,sf_email_inbound,
        channel_bank,channel_wirehouse,primary_prod_sold_fixed,
        sf_email_campaigns,advisor_hh_children,annuity_mkt_opp,
        advisor_advising_years,advisor_aum,advisor_annuity_selling_years,
        advisor_age,advisor_net_worth,advisor_credit_hist_mos,
        advisor_firm_changes,advisor_credit_score,wholesaler,region_ca,
        region_ny,region_fl,region_tx,region_ne,region_so,region_mw,
        sf_email_responses) then do;
   _LMR_BAD=1;
   goto _SKIP_000;
end;

*** Compute Linear Predictors;
drop _LP0;
_LP0 = 0;

*** Effect: sf_face_2_face;
_LP0 = _LP0 + (1.54806605556848) * sf_face_2_face;
*** Effect: sf_call_outbound;
_LP0 = _LP0 + (0.43292297652609) * sf_call_outbound;
*** Effect: sf_call_inbound;
_LP0 = _LP0 + (3.33361330275237) * sf_call_inbound;
*** Effect: sf_email_inbound;
_LP0 = _LP0 + (0.72972598206386) * sf_email_inbound;
*** Effect: channel_bank;
_LP0 = _LP0 + (0.13023630094943) * channel_bank;
*** Effect: channel_wirehouse;
_LP0 = _LP0 + (-0.74798124500567) * channel_wirehouse;
*** Effect: primary_prod_sold_fi;
_LP0 = _LP0 + (0.70498214051727) * primary_prod_sold_fixed;
*** Effect: sf_email_campaigns;
_LP0 = _LP0 + (0.71534338846528) * sf_email_campaigns;
*** Effect: advisor_hh_children;
_LP0 = _LP0 + (1.0802281916815) * advisor_hh_children;
*** Effect: annuity_mkt_opp;
_LP0 = _LP0 + (0.58882292191624) * annuity_mkt_opp;
*** Effect: advisor_advising_yea;
_LP0 = _LP0 + (-0.73073588108475) * advisor_advising_years;
*** Effect: advisor_aum;
_LP0 = _LP0 + (0.07506891087857) * advisor_aum;
*** Effect: advisor_annuity_sell;
_LP0 = _LP0 + (0.01422730389231) * advisor_annuity_selling_years;
*** Effect: advisor_age;
_LP0 = _LP0 + (-0.37516855573261) * advisor_age;
*** Effect: advisor_net_worth;
_LP0 = _LP0 + (4.65357491245134) * advisor_net_worth;
*** Effect: advisor_credit_hist_;
_LP0 = _LP0 + (0.80363357043123) * advisor_credit_hist_mos;
*** Effect: advisor_firm_changes;
_LP0 = _LP0 + (0.37051209634125) * advisor_firm_changes;
*** Effect: advisor_credit_score;
_LP0 = _LP0 + (1.27767507073579) * advisor_credit_score;
*** Effect: wholesaler;
_LP0 = _LP0 + (-0.0058210149538) * wholesaler;
*** Effect: region_ca;
_LP0 = _LP0 + (1.80322350412741) * region_ca;
*** Effect: region_ny;
_LP0 = _LP0 + (1.46163624666839) * region_ny;
*** Effect: region_fl;
_LP0 = _LP0 + (1.11248890496481) * region_fl;
*** Effect: region_tx;
_LP0 = _LP0 + (0.2244278452122) * region_tx;
*** Effect: region_ne;
_LP0 = _LP0 + (0.14650708257523) * region_ne;
*** Effect: region_so;
_LP0 = _LP0 + (-0.03094266653424) * region_so;
*** Effect: region_mw;
_LP0 = _LP0 + (-0.16189063433017) * region_mw;
*** Effect: sf_email_responses;
_LP0 = _LP0 + (0.54655840978808) * sf_email_responses;

*** Predicted values;
drop _MAXP _IY _P0 _P1;
_TEMP = -4.21901299577734  + _LP0;
if (_TEMP < 0) then do;
   _TEMP = exp(_TEMP);
   _P0 = _TEMP / (1 + _TEMP);
end;
else _P0 = 1 / (1 + exp(-_TEMP));
_P1 = 1.0 - _P0;
P_advisor_event_indicator0 = _P0;
_MAXP = _P0;
_IY = 1;
P_advisor_event_indicator1 = _P1;
if (_P1 >  _MAXP + 1E-8) then do;
   _MAXP = _P1;
   _IY = 2;
end;
select( _IY );
   when (1) do;
      I_advisor_event_indicator = '0' ;
      U_advisor_event_indicator = 0;
   end;
   when (2) do;
      I_advisor_event_indicator = '1' ;
      U_advisor_event_indicator = 1;
   end;
   otherwise do;
      I_advisor_event_indicator = '';
      U_advisor_event_indicator = .;
   end;
end;
_SKIP_000:
if _LMR_BAD = 1 then do;
I_advisor_event_indicator = '';
U_advisor_event_indicator = .;
P_advisor_event_indicator0 = .;
P_advisor_event_indicator1 = .;
end;
drop _TEMP;
