import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import PIL
import pandas as pd
import csv


L1 = {'categorie' : 1, 'values' : ['002','004', '007', '011', '012','015', '202','203','204','212','216','220','403','416','418','432','433','435'], 'name_eng' : 0, 'last_name_eng' : 0, 'name_tha' : 0, 'address_l1' : 0, 'address_l2': 0, 'bd_eng' : 0, 'bd_tha': 0, 'ed_eng': 0, 'ed_tha':0} 
L2 = {'categorie' : 2,'values' : ['028','031','035','040','041','207','213','215','225','226','402','408','412','413','417','421','431','434','436'], 'name_eng' : 0, 'last_name_eng' : 0, 'name_tha' : 0, 'address_l1' : 0, 'address_l2': 0, 'bd_eng' : 0, 'bd_tha': 0, 'ed_eng': 0, 'ed_tha':0} 
L3 = {'categorie' : 3,'values' : ['033','036','054','205','211','214','219','222','404','407','409','411','414','422','428','429'], 'name_eng' : 0, 'last_name_eng' : 0, 'name_tha' : 0, 'address_l1' : 0, 'address_l2': 0, 'bd_eng' : 0, 'bd_tha': 0, 'ed_eng': 0, 'ed_tha':0} 

liste_categories = [L1, L2, L3]

#Mean of White/Black pixels ratio for 3 categories


with open('Noise_Analysis_W&B_3.csv', 'w', newline='') as csvfile:
    fieldnames = ['categorie', 'name_eng', 'last_name_eng', 'name_tha', 'address_l1', 'address_l2', 'bd_eng', 'bd_tha', 'ed_eng', 'ed_tha']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    
    for liste in liste_categories :
        L = []
        parameter = ''
        compteur_name, compteur_date, compteur_address = 0,0,0
        for elmt in liste['values'] :
            if int(elmt) < 200 :
                compteur_name += 1
                parameter = 'name'
                L = ['name_eng', 'last_name_eng', 'name_tha']
            elif 200 <= int(elmt) < 400 :
                compteur_date += 1
                parameter = 'date'
                L = ['bd_eng', 'bd_tha', 'ed_eng', 'ed_tha']
            else : 
                compteur_address += 1
                parameter = 'address'
                L = ['address_l1', 'address_l2']
                
            for feature in L :
                path = '/Users/GeoffroyDeMeyer/Desktop/Thai_Card/cropped/' + str(parameter) +'/'+str(elmt) + '_cropped_rot_' + str(feature) +'.TIFF'
                img = PIL.Image.open(str(path))
                r = ratio(img)
                liste[str(feature)] += r
                
        if compteur_name != 0:      
            liste['name_eng'] = liste['name_eng'] / compteur_name
            liste['last_name_eng'] = liste['last_name_eng'] / compteur_name
            liste['name_tha'] = liste['name_tha'] / compteur_name
        if compteur_date != 0:
            liste['bd_eng'] = liste['bd_eng'] / compteur_date
            liste['bd_tha'] = liste['bd_tha'] / compteur_date
            liste['ed_eng'] = liste['ed_eng'] / compteur_date
            liste['ed_tha'] = liste['ed_tha'] / compteur_date
        if compteur_address != 0:
            liste['address_l1'] = liste['address_l1'] / compteur_address
            liste['address_l2'] = liste['address_l2'] / compteur_address
        
        for elmt in fieldnames[1:]:
            liste[str(elmt)] = round(liste[str(elmt)])
                
        liste.pop('values')
        writer.writerow(liste)
        
        
#Doc containing the mean value for each category and each box

doc = pd.read_csv('Noise_Analysis_W&B_3.csv')

#predicting the appartenance to a category or not

parameters = ['name', 'address', 'date']

for parameter in parameters :
    for root, dirs, filenames in os.walk('/Users/GeoffroyDeMeyer/Desktop/Thai_Card/cropped/' + str(parameter)):
        for image in filenames :
            img = PIL.Image.open(str(root) + '/' + str(image))
            white, black = 0,0
            for pixel in img.getdata():
                if pixel == 0 : 
                    black += 1
                elif pixel == 255 :
                    white += 1
            ratio = white/black
            minimum = abs(ratio-doc[str(str(image)[16:-5])].iloc[0])
            categorie = 1
            for i in range(0, len(doc)) :
                value = abs(ratio-doc[str(str(image)[16:-5])].iloc[i])
                if  value < minimum :
                    minimum = value
                    categorie = i+1
            img.save('/Users/GeoffroyDeMeyer/Desktop/Thai_Card/cropped_split/' + str(len(liste_categories))+ '_cat/cat_'+ str(categorie) + '/' + str(image))
                
                
            