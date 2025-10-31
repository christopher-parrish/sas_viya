   /*---------------------------------------------------------
     Generated SAS Scoring Code
     Date: 16Jul2025:20:55:14
     -------------------------------------------------------*/

   drop _badval_ _linp_ _temp_ _i_ _j_;
   _badval_ = 0;
   _linp_   = 0;
   _temp_   = 0;
   _i_      = 0;
   _j_      = 0;
   drop MACLOGBIG;
   MACLOGBIG= 7.0978271289338392e+02;

   array _xrow_0_0_{11} _temporary_;
   array _beta_0_0_{11} _temporary_ (    6.13014507386307
          -2.27986659354005
          -1.29942182078289
          -0.94832987161269
          -1.60952416684927
           -1.5542528633564
           1.92032705770585
           1.94492301645574
          -0.40692908420976
           -0.0411524862399
           0.00679165073037);

   if missing(prior_ctr_indicator)
      or missing(address_change_2x_indicator)
      or missing(distance_to_bank)
      or missing(in_person_contact_indicator)
      or missing(distance_to_employer)
      or missing(linkedin_indicator)
      or missing(citizenship_country_risk)
      or missing(checking_only_indicator)
      or missing(cross_border_trx_indicator)
      or missing(marital_status_single)
      then do;
         _badval_ = 1;
         goto skip_0_0;
   end;

   do _i_=1 to 11; _xrow_0_0_{_i_} = 0; end;

   _xrow_0_0_[1] = 1;

   _xrow_0_0_[2] = marital_status_single;

   _xrow_0_0_[3] = checking_only_indicator;

   _xrow_0_0_[4] = prior_ctr_indicator;

   _xrow_0_0_[5] = address_change_2x_indicator;

   _xrow_0_0_[6] = cross_border_trx_indicator;

   _xrow_0_0_[7] = in_person_contact_indicator;

   _xrow_0_0_[8] = linkedin_indicator;

   _xrow_0_0_[9] = citizenship_country_risk;

   _xrow_0_0_[10] = distance_to_employer;

   _xrow_0_0_[11] = distance_to_bank;

   do _i_=1 to 11;
      _linp_ + _xrow_0_0_{_i_} * _beta_0_0_{_i_};
   end;

   skip_0_0:
   length I_ml_indicator $12;
   label I_ml_indicator = 'Into: ml_indicator';
   array _levels_0_{2} $ 12 _TEMPORARY_ ('0'
   ,'1'
   );
   label P_ml_indicator = 'Predicted: ml_indicator';
   if (_badval_ eq 0) and not missing(_linp_) then do;
      if (_linp_ > 0) then do;
         P_ml_indicator = 1 / (1+exp(-_linp_));
      end; else do;
         P_ml_indicator = exp(_linp_) / (1+exp(_linp_));
      end;
      if P_ml_indicator >= 0.5                  then do;
         I_ml_indicator = _levels_0_{1};
      end; else do;
         I_ml_indicator = _levels_0_{2};
      end;
   end; else do;
      _linp_ = .;
      P_ml_indicator = .;
      I_ml_indicator = ' ';
   end;


