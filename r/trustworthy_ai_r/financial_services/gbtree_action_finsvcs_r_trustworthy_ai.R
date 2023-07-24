cas.fairAITools.mitigateBias(conn,
                             biasMetric='DEMOGRAPHICPARITY',
                             event='1',
                             learningRate='0.01',
                             maxIters='10',
                             predictedVariables=c('P_BAD0', 'P_BAD1'),
                             response='BAD',
                             responseLevels=c('0', '1'),
                             sensitiveVariable='reason',
                             table='hmeq',
                             tolerance='0.005',
                             trainProgram='
         decisionTree.gbtreeTrain result=train_res /
            table=table,
            weight=weight,
            target="BAD",
            inputs= {
               "loan", "mortdue", "value",
               "yoj", "derog", "delinq",
               "clage", "ninq", "clno",
               "debtinc", "job"
            },
            nominals={"BAD","job"},
            nBins=50,
            quantileBin=True,
            maxLevel=5,
            maxBranch=2,
            leafSize=5,
            missing="USEINSEARCH",
            minUseInSearch=1,
            binOrder=True,
            varImp=True,
            mergeBin=True,
            encodeName=True,
            nTree=15,
            seed=12345,
            ridge=1,
            savestate={
               name="hmeq_gb_astore",
               replace=True
            }
         ;
         astore.score result=score_res /
            table=table,
            casout=casout,
            copyVars=copyVars,
            rstore="hmeq_gb_astore"
         ;
      ',
                             tuneBound='True')




cas.fairAITools.mitigateBias(conn,
                             biasMetric='DEMOGRAPHICPARITY',
                             event='1',
                             learningRate='0.01',
                             maxIters='10',
                             predictedVariables=c('P_event_indicator0', 'P_event_indicator1'),
                             response='event_indicator',
                             responseLevels=c('0', '1'),
                             sensitiveVariable='gender',
                             table='financial_services_prep',
                             tolerance='0.005',
                             tuneBound='True',
                             trainProgram='
                                 decisionTree.gbtreeTrain result=train_res /
                                    table=table,
                                    weight=weight,
                                    target="event_indicator",
                                    inputs= {
                                       "at_current_job_1_year", "num_dependents",
                                       "age", "amount", "credit_history_mos", "credit_score",
                                       "debt_to_income", "net_worth", "num_transactions"
                                    },
                                    nominals={"event_indicator"},
                                    nBins=50,
                                    quantileBin=True,
                                    maxLevel=5,
                                    maxBranch=2,
                                    leafSize=5,
                                    missing="USEINSEARCH",
                                    minUseInSearch=1,
                                    binOrder=True,
                                    varImp=True,
                                    mergeBin=True,
                                    encodeName=True,
                                    nTree=15,
                                    seed=12345,
                                    ridge=1,
                                    savestate={
                                       name="finsvcs_gb_astore",
                                       replace=True
                                    }
                                 ;
                                 astore.score result=score_res /
                                    table=table,
                                    casout=casout,
                                    copyVars=copyVars,
                                    rstore="finsvcs_gb_astore"
                                 ;
                              ')

results <- cas.table.fetch(conn, table=list(caslib="casuser", name="score_res")) 
results