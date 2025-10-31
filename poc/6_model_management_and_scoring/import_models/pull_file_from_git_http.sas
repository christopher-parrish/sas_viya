filename gitfile "/innovationlab-export/innovationlab/homes/Chris.Parrish@sas.com/logit_sas_amlbank_git.sas"; 
*filename gitfile temp; /* if prefer not to keep file */

proc http url="https://raw.githubusercontent.com/christopher-parrish/sas_viya/refs/heads/main/sas/logit_sas/logit_ds1code_score"
          out=gitfile;  
run;
