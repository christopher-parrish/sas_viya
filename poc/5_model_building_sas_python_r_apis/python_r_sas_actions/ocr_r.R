
########################################################
###  Optical Character Recognition (OCR) Extraction  ###
########################################################

###################
### Credentials ###
###################

library(askpass)
library(sys)

username <- askpass("USERNAME")
password <- askpass("PASSWORD")
wd <- askpass("What is the Working Directory for this R Session?")
source(file.path(wd, 'password.r'))

###################
### Environment ###
###################

library(swat)

conn <- swat::CAS(hostname=hostname, port=port, username, password, protocol=protocol)
print(cas.builtins.serverStatus(conn))

###################
### Read Images ###
###################

library(tesseract)

eng <- tesseract("eng")
img_location <- ".../git/sas_viya/data/images/credit_application/credit_app_1.png"
text <- tesseract::ocr(img_location, engine=eng)
cat(text)