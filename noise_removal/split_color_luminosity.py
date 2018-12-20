import matplotlib.pyplot as plt
import numpy as np
import os
import PIL

folder = ['/Users/GeoffroyDeMeyer/Desktop/Thai_Card/MTL/ID Card File/S1_Name_Train','/Users/GeoffroyDeMeyer/Desktop/Thai_Card/MTL/ID Card File/S2_Date_Train', '/Users/GeoffroyDeMeyer/Desktop/Thai_Card/MTL/ID Card File/S3_Address_Train']

for path in folder :
    for root, dirs, filenames in os.walk(str(path)):
        filenames.pop(filenames.index('.DS_Store'))
        for image in filenames :
            img = PIL.Image.open(str(root) + '/' + str(image))
            if len(img.getextrema()) == 3 :
                gray_img = img.convert('L')
                img = gray_img
            img_array = np.array(img)
            L = img_array.flatten()
            median = np.median(L)
            if median <= 125 :
                img.save(str(path).replace('ID Card File', 'ID_Card_File_split') + '/Underexposed'+ '/' + str(image))
            else :
                img.save(str(path).replace('ID Card File', 'ID_Card_File_split') + '/Overexposed'+ '/' + str(image))

