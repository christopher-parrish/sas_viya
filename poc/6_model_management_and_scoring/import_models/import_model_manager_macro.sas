cas casauto sessopts=(caslib=public, metrics=true, timeout=900);

filename modelref '/innovationlab-export/innovationlab/homes/Chris.Parrish@sas.com/logit_sas_amlbank_git.sas';
%mm_import_model(
    trainTable       = public.aml_bank_prep,
    target           = ml_indicator,
    targetLevel      = Binary,
    modelnm          = logit_sas_amlbank_git,
    modeldesc        = Logistic Regression,
    modelfunc        = Classification,
    modelloc         = modelref,
    algorithm        = %nrstr(Regression),
    modeler          = cp,
    scorecodetype    = DATASTEP,
    traincodetype    = DATASTEP,
    fileType         = DATASTEP,
    filesizeoverride = N,
    projectID        = %str(1b3b8624-4b88-4735-b70c-d2bf255ecf0d),
    importinto       = project,
    modelID          = myModelID
);
%put &myModelID;