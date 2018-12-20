#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
"""
    Extract the dates of the field images in AD and Buddist year.
"""

from PIL import Image
import pytesseract
from scipy.ndimage import interpolation as inter
import numpy as np
from matplotlib import pyplot as plt
import os
from os import listdir
from os.path import isfile, join
from difflib import SequenceMatcher
import collections
import pandas as pd
import re

# scenario = ["name", "date", "address"]
# cropped_img_path = '/home/tania/Thai_id/cropped_v2/'

# s = scenario[1]
# onlyfiles = [f for f in listdir(cropped_img_path+s+"/") if isfile(join(cropped_img_path+s+"/", f))]
# onlyfiles[:5]

tha_months = ["ม.ค.", "มี.ค.", "เม.ย.", "พ.ค.", "มิ.ย.", "ก.ค.", "ส.ค.", "ก.ย.", "ต.ค.", "พ.ย.", "ธ.ค."]
eng_months = ["Jan.", "Feb.", "Mar.", "Apr.", "May.", "Jun.", "Jul.", "Aug.", "Sep.", "Oct.", "Nov.", "Dec."]

def eng_month_matching(eng_months, text,eng_month_ratio_threshold):
    """Match English month."""    
    # calculate similarity between candidate and titles
    month = [(t, similar(text.lower(), t.replace(".","").lower())) for t in eng_months]
    # sort by the highest similarity
    max_ratio_month = sorted(month, key=lambda tup: -tup[1])
    # print(max_ratio_month)
    # check if ratio is higher then min
    if max_ratio_month[0][1] >= eng_month_ratio_threshold:
        result_month = max_ratio_month[0][0]
    else:
        result_month = ""
    return result_month

def date_matching(text,name_ratio_threshold):
    """Match english name."""
    
    # check similarity
    if similar("date", text.lower()) >= name_ratio_threshold:
        return "Date"
    else:
        return ""

def birth_matching(text,name_ratio_threshold):
    """Match english name."""
    
    # check similarity
    if similar("birth", text.lower()) >= name_ratio_threshold:
        return "Birth"
    else:
        return ""

def clean_date(text):
    """Remove punctuations from extracted name."""
    # replace some punctuations by space
    text = text.replace("/", " ").replace("\"", " ").replace("%", "3")
    # remove punctuation
    punctuations = "!@#$%^&*()_+=,./<>\|™;~:©®\[\]\{\}“\"\‘\'\’"
    cleaned_text = ''.join(ch for ch in text if ch not in punctuations)
    return cleaned_text

def extract_date_modes_balancing(img, step,lang,config_p_eng_8,config_p_eng_12, ranges):
    # extract text with different modes
    text_12, freq_12 = extract_date(img, step,lang,config_p_eng_12, ranges)
    text_8, freq_8 = extract_date(img, step,lang,config_p_eng_8, ranges)
    
    # try to balance with 2 modes ???
    if freq_12 >= freq_8:
        if freq_12 > 4:
            text = text_12
        else: 
            text = text_8
    else:
        text = text_8
        
    return text

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def extract_date(img, step,  lang,config_p, ranges):
    # get image size
    w, h = img.size
    # print("Size", (w,h))
    # define variables
    freqs = {}
    max_freq = ("", 0)
    
    # iterate through size
    for i in ranges:
        # size values
        wb, hb = int((1 + i*step) * w), int((1 + i*step) * h)
        
        # text extraction
        txt = pytesseract.image_to_string(img.resize((wb, hb)), lang, config=config_p)
        
        # count frequency only last word
        txt_last=txt
        # find the most frequent result
        if txt_last != "" and not "\n" in txt_last:
            if txt_last in freqs:
                freqs[txt_last] += 1
            else:
                freqs[txt_last] = 1

            max_freq = max_freq if max_freq[1] > freqs[txt_last] else (txt_last, freqs[txt_last])
        
            print(txt)
            
    result = clean_date(max_freq[0])
    
    print("====================")
    print("Result = ", result)
    
    return result, max_freq[1] 

def get_month_from_day(text, day,eng_months,eng_month_ratio_threshold):
    "Find the month from date position"
    # find date first and end position
    start_index = text.find(day)
    end_index = start_index + len(day)
    # get month
    month = text[(end_index+1):].split(" ")[0]
    # print(month)
    return  eng_month_matching(eng_months, month,eng_month_ratio_threshold)

def bd_year_validator(year):
    if int(year) >= bd_year[0] and int(year) <= bd_year[1]:
        pass
    else:
        if year[1] == '0':
            year = year[0] + '9' + year[2:]
    return year

def find_bd_date(text,bd_date_ratio_threshold,eng_month_ratio_threshold,eng_months):
    """Find the title and name"""
    words = text.split(" ")
    words = [w for w in words if w != ""]
    # standart situation when there are 3 words
    if len(words) == 3: 
        # check if the words Date and Birth exist
        try:
            year = re.match(r'.*([1-3][0-9]{3})', text).group(1)
        except: 
            year = re.match(r'\d+', text).group(1)
        try:
            day = re.match(r'([0-9][0-9]{0,1}) ', text).group(1).replace(" ", "")
        except: 
            day = "-"
        try:
            month = get_month_from_day(text, day,eng_months,eng_month_ratio_threshold)
        except: 
            month = "-"
    # if there are 5 words 
    elif  len(words) == 1: 
        year = re.match(r'.*([1-3][0-9]{3})', text).group(1)
        month = "-"
        day = " - "        
    # if there are 2 words 
    elif  len(words) == 2: 
        # extract year
        year = re.match(r'.*([1-3][0-9]{3})', text).group(1)  
        # match the date merged with month text
        day = re.match(r'([0-9][0-9]{0,1})\w+', text).group(1).replace(" ", "")
        # subtract day text
        month_text = text.replace(day, "").split(" ")[0]
        month = eng_month_matching(eng_months, month_text,eng_month_ratio_threshold)
    else:
        day =  re.match(r'([0-9][0-9]{0,1}) ', text).group(1).replace(" ", "")
        month = get_month_from_day(text, day,eng_months,eng_month_ratio_threshold)
        year = re.match(r'.*([1-3][0-9]{3})', text).group(1)    
        
    return day, month, bd_year_validator(year)

fields = dict()
fields["name"] = "name_eng"
fields["name"] = "last_name_eng"
fields["name"] = "name_tha"

config_p_eng_12="--psm 12 --oem 1"
config_p_eng_8="--psm 8 --oem 1"

config_p_tha="--psm 4 --oem 0"

step = 0.2
ranges=range(20, 50)
ranges_tha = range(25,35)
ranges_date_eng = range(5,50)

name_ratio_threshold=0.25
last_name_ratio_threshold=0.25
title_ratio_threshold=0.3
tha_name_ratio_threshold=0.15
tha_title_ratio_threshold=0.25
bd_date_ratio_threshold = 0.45
eng_month_ratio_threshold=0.15

bd_year = (1900, 2020)
bd_files = [x for x in onlyfiles if "bd_eng" in x]

# read the image
lang = "eng"
img = Image.open(cropped_img_path +s+"/"+bd_files[10])

# find the best output
text = extract_date_modes_balancing(img, step, lang, config_p_eng_12, config_p_eng_12, ranges_date_eng)
print("=============")
print("Text = ", text)
    
# split by words
day, month, year =  find_bd_date(text, bd_date_ratio_threshold, eng_month_ratio_threshold, eng_months)
print("========================================")
print("Day = ", day)
print("Month = ", month)
print("Year = ", year)

text = '19Mar 1968'
day = re.match(r'([0-9][0-9]{0,1})\w+', text).group(1).replace(" ", "")

month_text = text.replace(day, "")

(w, h) = img.size
print("Image size = ", (w,h))
step = 0.2

for i in range(10, 50):
    print(i,"*",step," = ",(int(w*(i*step)), int(h*(i*step))), " resize =>", pytesseract.image_to_string(img.resize((int(w*(i*step)), int(h*(i*step)))), "tha", config="--psm 5 --oem 1"))

