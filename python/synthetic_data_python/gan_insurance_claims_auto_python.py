###########################
###  Train a GAN Model  ###
###########################

###################
### Credentials ###
###################

import runpy
import os

from password import hostname, session, protocol, port, wd, token_dir, token_pem, token, token_refresh

runpy.run_path(path_name='password.py')

###################
### Environment ###
###################

import swat
from casauth import CASAuth

access_token = open(token, "r").read()
conn = CASAuth(token_dir, ssl_ca_list=token_pem)
print(conn.serverstatus())

conn.session.timeout(time=3600)

###################
###  GAN Model  ###
###################

### import actionsets
conn.loadactionset('generativeAdversarialNet')

results = conn.generativeAdversarialNet.tabularGanTrain(
        table = dict(caslib = "casuser", name = "insurance_claims_auto",
                     vars = ["claim_indicator",
							 "new_driver",
							 "income"]
                             ),
         nominals        = ["claim_indicator",
                            "new_driver"],
		 gmmOptions		 = dict(alpha = 1, maxClusters = 10, seed = 42,
							inference = dict(covariance = 'DIAGONAL',
    										 maxVbIter = 30,
                                             threshold = 0.01)),
         gpu             = dict(useGPU = False, device = 0),
         optimizerAe     = dict(method = "ADAM", numEpochs = 3),
         optimizerGan    = dict(method = "ADAM", numEpochs = 5),
         seed            = 12345,
         scoreSeed       = 0,
         numSamples      = 1000,
         saveState       = dict(name = "insurance_claims_synth_astore", replace = True),
         casOut          = dict(name = "insurance_claims_synth_out", replace = True)
         )

conn.table.fetch(table = dict(caslib = "casuser", name = "insurance_claims_synth_out"))

