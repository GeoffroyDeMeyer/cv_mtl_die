
#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
"""
    Crop the image into different fields from the ID Card.
"""

from PIL import Image
import collections
from os import listdir
from os.path import isfile, join
import numpy as np

def bd_extractor(rotated_img, bd_ratio, area_type):
    """Crop date of birth from rotated image"""

    # convert CV2 to PIL
    rotated_img = Image.fromarray(rotated_img)
    # rotated_img.show()
    # get the size
    (w, h) = rotated_img.size    
    # get the area for cropping
    area =  (w*bd_ratio[0]/100.0, h*bd_ratio[1]/100.0,w*bd_ratio[2]/100.0,h*bd_ratio[3]/100.0)
    # crop image
    cropped_img =rotated_img.crop(area)    

    return cropped_img

def area_cropping(rotated_img, area_ratio, area_type):
    """Crop date of birth from rotated image"""
    
    # convert CV2 to PIL
    rotated_img = Image.fromarray(rotated_img)
    # get the size
    (w, h) = rotated_img.size    
    # get the area for cropping
    area =  (w*area_ratio[0]/100.0, h*area_ratio[1]/100.0,w*area_ratio[2]/100.0,h*area_ratio[3]/100.0)
    # crop image
    cropped_img =rotated_img.crop(area)    

    return cropped_img

# define the ratio for each field
date_area_ratio=dict()
name_area_ratio=dict()
address_area_ratio=dict()

date_area_ratio["bd_eng"] = (46.2, 45.0, 75.0, 63.8)
date_area_ratio["bd_tha"] = (40.86, 38.85, 66.0, 53.5)
date_area_ratio["ed_tha"] = (52.8, 73.06, 73.75, 84.5)
date_area_ratio["ed_eng"] = (52.8, 82.5, 73.75, 93.0)

name_area_ratio["name_tha"] =(24.7, 16.06, 92.75, 30.5)
name_area_ratio["name_eng"] =(36.5, 25.06, 94.75, 37.5)
name_area_ratio["last_name_eng"] =(42.9, 31.7, 95.75, 45.5)

address_area_ratio["address_l1"] =(13.5, 60.06, 68.75, 72.0)
address_area_ratio["address_l2"] =(8.0, 68.6, 58.75, 76.5)



# scenario = ["name", "date", "address"]
# ratated_img_path = '/home/tania/Thai_id/rotated_cropped/'
# cropped_BD_path = '/home/tania/Thai_id/cropped/'

def extract_fields(image, field):
    """Extract image field boxes for input image (ID Card)."""
    
    images_name_field_boxes = collections.OrderedDict([
        ('name_eng', ''),
        ('name_tha', ''),
        ('last_name_eng', '')
    ])
    images_date_field_boxes = collections.OrderedDict([
        ('bd_eng', ''),
        ('bd_tha', ''),
        ('ed_eng', ''),
        ('ed_tha', '')
    ])
    images_address_field_boxes = collections.OrderedDict([
        ('address_l1', ''),
        ('address_l2', '')
    ])


    if field == 'Name':
        # process name fields
        for k,v in name_area_ratio.items():
            images_name_field_boxes[k] = bd_extractor(image, v, k)

        return images_name_field_boxes

    elif field == 'Date':
        # process date fields
        for k,v in date_area_ratio.items():
            images_date_field_boxes[k] = bd_extractor(image, v, k)

        return images_date_field_boxes

    elif field == 'Address':
        # process address fields
        for k,v in address_area_ratio.items():
            images_address_field_boxes[k] = bd_extractor(image, v, k)

        return images_address_field_boxes

# if __name__ == "__main__":
#     # iterate through scenario
#     for s in scenario:
#             # list the folder with images
#             onlyfiles = [f for f in listdir(ratated_img_path+s+"/") if isfile(join(ratated_img_path+s+"/", f))]
#             # iterate through images
#             if s == "name":
#                 for i in range(len(onlyfiles)):
#                     # iterate through areas                
#                     for k,v in name_area_ratio.items():
#                         bd_extractor(ratated_img_path+s+"/"+onlyfiles[i],cropped_BD_path+s+"/"+onlyfiles[i], v, k)
#             elif s == "date":
#                 for i in range(len(onlyfiles)):
#                     # iterate through areas                
#                     for k,v in date_area_ratio.items():
#                         bd_extractor(ratated_img_path+s+"/"+onlyfiles[i],cropped_BD_path+s+"/"+onlyfiles[i], v, k)
#             else:
#                 for i in range(len(onlyfiles)):
#                     # iterate through areas                
#                     for k,v in address_area_ratio.items():
#                         bd_extractor(ratated_img_path+s+"/"+onlyfiles[i],cropped_BD_path+s+"/"+onlyfiles[i], v, k)
