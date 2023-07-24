
from PIL import Image

import pytesseract

#img_location = "/export/home/chparr/casuser/git/data/images/credit_application/credit_app_1.png"
img_location = "C:/Users/chparr/OneDrive - SAS/git/sas_viya/data/images/credit_application/credit_app_1.png"
print(pytesseract.image_to_string(Image.open(img_location)))