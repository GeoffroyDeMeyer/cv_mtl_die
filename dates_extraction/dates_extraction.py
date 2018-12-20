from PIL import Image
import pytesseract
from scipy.ndimage import interpolation as inter

import numpy as np

import os
import collections

from os import listdir
from os.path import isfile, join
from difflib import SequenceMatcher

import pandas as pd
import re

import random
import argparse

DEBUG = False
#DEBUG = True

FIX_TYPOS          = False
ED_TO_BD           = False
ADJUST_LIFELONG_FP = False
ADJUST_LIFELONG_FN = True
FILTERED_YEARS     = True

ED_TO_BD_THRES     = 2.5
LIFELONG_FP_THRES  = 4
LIFELONG_FN_THRES  = 40

UNC_LL_YEAR   = [1948, 1968] # range of people who may or may not have a LIFE LONG expiration date

# Changing these does not completely redefine the range!
RANGE_YEAR = {'ed': {'eng': [2000, 2030], 'tha': [2543, 2573]},
              'bd': {'eng': [1930, 2010], 'tha': [2473, 2553]}}


def log(*args):
    if DEBUG:
        print(''.join(map(str,args)))

class DateExtractor:
    def __init__(self):
        # self.model1 = {'eng' : '--psm 12 --oem 1', 'tha' : '--psm 4 --oem 1'}
        # self.model2 = {'eng' : '--psm 8 --oem 1', 'tha' : '--psm 8 --oem 1'}
        self.ranges_date = {'eng': range(5,50), 'tha' : range(5, 50)}

        self.step = 0.2

        self.bd_date_ratio_threshold = 0.45
        self.month_ratio_threshold = {'eng' : 0.15, 'tha' : 0.15}
        self.months = {'eng' : ["Jan.", "Feb.", "Mar.", "Apr.", "May.", "Jun.", "Jul.", "Aug.", "Sep.", "Oct.", "Nov.", "Dec."],
                       'tha' : ["ม.ค.", "ก.พ.", "มี.ค.", "เม.ย.", "พ.ค.", "มิ.ย.", "ก.ค.", "ส.ค.", "ก.ย.", "ต.ค.", "พ.ย.", "ธ.ค."]}
        self.months_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        self.digits = '0123456789'
        self.lifelong_text = {'eng' : 'LIFE LONG', 'tha' : 'ตลอดชีพ'}
        self.lifelong_similarity_threshold = {'eng' : 0.25, 'tha' : 0.3}

        self.correlation_cen_eng = {'0' : '0', '1' : '9', '2' : '9', '3' : '9', '4' : '9', '5' : '9', '6' : '0', '7' : '9', '8' : '0', '9' : '9'}
        self.correlation_cen_tha = {'0' : '5', '1' : '4', '2' : '5', '3' : '5', '4' : '4', '5' : '5', '6' : '5', '7' : '4', '8' : '5', '9' : '5'}
        self.correlation_mil = {'0' : '0', '1' : '1', '2' : '2', '3' : '2', '4' : '1', '5' : '2', '6' : '1', '7' : '1', '8' : '2', '9' : '2'}
        self.correlation_day0 = {'-' : '', '' : '', '0' : '0', '1' : '1', '2' : '2', '3' : '3', '4' : '1', '5' : '2', '6' : '3', '7' : '2', '8' : '3', '9' : '3'}
        self.correlation_day1 = {'0' : '0', '1' : '1', '2' : '1', '3' : '1', '4' : '1', '5' : '0', '6' : '0', '7' : '1', '8' : '0', '9' : '0'}

    def isdigit(self, val):
        for c in val:
            if c not in self.digits:
                return False
        return True

    def month_matching(self, eng_months, text,eng_month_ratio_threshold):
        """Match English month"""
        # calculate similarity between candidate and titles
        month = [(t, self.similar(text, t.replace(".",""))) for t in eng_months]
        # sort by the highest similarity
        max_ratio_month = sorted(month, key=lambda tup: -tup[1])
        # check if ratio is higher then min
        if max_ratio_month[0][1] >= eng_month_ratio_threshold:
            result_month = max_ratio_month[0][0]
        else:
            result_month = ""
        return result_month


    def clean_date(self, text, lang):
        """Remove punctuations from extracted name"""
        # replace some punctuations by space
        text = text.replace("/", " ").replace("\"", " ").replace("%", "3")
        # remove punctuation
        if lang == 'eng':
            punctuations = "!@#$%^&*()_+=,.-/<>\|™;~:©®\[\]\{\}“\"\‘\'\’"
        else:
            punctuations = "0123456789!@#$%^&*()_+=-/<>\|™;~:©®\[\]\{\}“\"\‘\'\’"
        cleaned_text = ''.join(ch for ch in text if ch not in punctuations)
        if lang == 'tha':
            cleaned_text = cleaned_text.replace(',', '.')
        return cleaned_text

    def choose_higher(self, value1, freq1, value2, freq2):
        if freq1 >= freq2:
            value = value1
            freq = freq1
        else:
            if freq2 > 4:
                value = value2
                freq = freq2
            else:
                value = value1
                freq = freq1
        return value, freq

    def attempt_day(self, candidate):
        current_try = ''
        for i in range(len(candidate) - 1, -1, -1):
            c = candidate[i]
            if c in self.digits:
                current_try = c + current_try
            if len(current_try) == 2:
                break
        return current_try

    def extract_date_for_model(self, img, lang, model, range_scale, date_type='bd'):
        w, h = img.size
        txts = []
        log('Extracting for language {} and model {}'.format(lang, model))
        for i in range_scale:
            wb, hb = int(i * w), int(i * h)
            # text extraction
            txt = pytesseract.image_to_string(img.resize((wb, hb)), lang, config=model)
            txts.append(txt.replace('\n', ' '))
            log('" {} "'.format(txt.replace('\n', ' ')))
        return txts

    def get_date_parts(self, txt, lang, date_type):
        # count frequency only last word
        txt_last=txt
        # find the most frequent result
        if txt_last != "" and not "\n" in txt_last:
            parts = txt_last.split()
            for part in parts:
                similarity = self.similar(self.lifelong_text[lang], txt_last)
                if date_type == 'ed':
                    if similarity > self.lifelong_similarity_threshold[lang]:
                        log('Detected Lifelong with {}'.format(txt))
                        return self.lifelong_text[lang]

            index = -1
            year = ''.join(c for c in parts[index] if c in self.digits)
            old_year = parts[index]
            while (len(parts) + index >= 1) and ((not self.isdigit(year)) or (len(year) != 4)):
                index -= 1
                year = ''.join(c for c in parts[index] if c in self.digits)
                old_year = parts[index]
            if index == -len(parts):
                index += 1
                txt_last = ' '.join(parts[:-1])
                year = ''.join(c for c in parts[-1] if c in self.digits)
                old_year = parts[-1]
            else:
                txt_last = ' '.join(parts[:index])

            parts = txt_last.split()
            index = 0
            if len(parts) > 0:
                #day = self.attempt_day(parts[index])
                day = 'x'
                index = 0
                for part in parts:
                    day = self.attempt_day(''.join(parts[:index+1]))
                    if day.isdigit() and len(day) > 2:
                        break
                    index += 1
                #while (index != len(parts)) and ((not day.isdigit()) or (len(day) > 2)):
                #    day = self.attempt_day(parts[index])
                #    index += 1
                if index == len(parts):
                    month_start = 0 if len(parts) == 1 else 1
                    if lang == 'tha':
                        month = self.clean_date(''.join(parts[month_start:]), lang)
                    else:
                        month = ''.join(''.join(c for c in ''.join(parts[month_start:]) if c.isalpha()))
                    day = self.attempt_day(''.join(parts[0:]))
                else:
                    if index == (len(parts) - 1):
                        month = ''.join(''.join(c for c in parts[index] if c.isalpha()))
                    else:
                        month = ''.join(parts[index + 1:])
            else:
                day = '-'
                month = '-'

            if month == '':
                month = ''.join(c for c in old_year if c.isalpha())
            if month == '':
                month = '-'
            if month != '-':
                month = self.month_matching(self.months[lang], month, self.month_ratio_threshold[lang])
            log('Detected "{}" "{}" "{}"'.format(day, month, year))
            return day, month, year
        else:
            return ''

    def extract_date_modes_balancing(self, img, lang, models, range_scale, date_type='bd'):
        ######
        txts = []
        days0 = {}
        days1 = {}
        months = {}
        millenia = {}
        centuries = {}
        tens = {}
        years = {}
        max_ten = ['', 0]
        max_yr = ['', 0]
        max_cen = ['', 0]
        max_mil = ['', 0]
        max_day0 = ['', 0]
        max_day1 = ['', 0]
        max_month = ['', 0]
        lifelong = 0
        total_count_year = 0
        total_count_day = 0
        total_count_month = 0
        total_count = 0
        detected_lifelong = False

        for model in models:
            txts.extend(self.extract_date_for_model(img, lang, model, range_scale, date_type))

        txts = [txt.replace('\n', '') for txt in txts]

        for txt in txts:
            if txt == '':
                continue
            total_count += 1
            returned = self.get_date_parts(txt, lang, date_type)
            if returned == '':
                continue
            if returned == self.lifelong_text[lang]:
                lifelong += 1
            else:
                day, month, year = returned
                if self.isdigit(year) and len(year) == 4:
                    total_count_year += 1
                    mil = year[0]
                    cen = year[1]
                    ten = year[2]
                    yr = year[3]
                    max_cen = self.update_freq(cen, centuries, max_cen)
                    max_mil = self.update_freq(mil, millenia, max_mil)
                    max_yr = self.update_freq(yr, years, max_yr)
                    max_ten = self.update_freq(ten, tens, max_ten)

                if day != '' and self.isdigit(day) and len(day) <= 2:
                    total_count_day += 1
                    if len(day) == 2:
                        max_day0 = self.update_freq(day[0], days0, max_day0)
                        max_day1 = self.update_freq(day[1], days1, max_day1)
                    else:
                        max_day0 = self.update_freq(day, days0, max_day0)
                        #max_day1 = self.update_freq('', days1, max_day1)
                #else:
                    #max_day0 = self.update_freq('-', days0, max_day0)

                if FIX_TYPOS and (month != '' and month != '-'):
                    max_month = self.update_freq(month, months, max_month)
                    total_count_month += 1
                elif not FIX_TYPOS and (month != '' or month != '-'):
                    max_month = self.update_freq(month, months, max_month)
                    total_count_month += 1

        log('FREQUENCIES FOR DAY \n{} \n{}'.format(days0, days1))
        log('FREQUENCIES FOR MONTH \n{}'.format(months))
        log('FREQUENCIES FOR YEAR \n{} \n{} \n{} \n{}'.format(millenia, centuries, tens, years))
        log('LIFELONG {} TOTAL {}'.format(lifelong, total_count))
        if (total_count != 0 and lifelong >= (total_count * 0.2)):
            detected_lifelong = True

        final_first_part = ''

        if lang == 'eng':
            final_mil = self.correlation_mil[max_mil[0]] if max_mil[0] != '' else ('1' if date_type == 'bd' else '2')
            final_cen = self.correlation_cen_eng[max_cen[0]] if max_cen[0] != '' else ('9' if date_type == 'bd' else '0')
        else:
            final_mil = '2'
            final_cen = self.correlation_cen_tha[max_cen[0]] if max_cen[0] != '' else '5'

        if lang == 'eng':
            if (final_mil == '1' and final_cen == '0') or (final_mil == '2' and final_cen == '9'):
                if date_type == 'bd':
                    final_mil = '1'
                    final_cen = '9'
                else:
                    final_mil = '2'
                    final_cen = '0'
                max_mil[1] = 0
                max_cen[1] = 0

        if lang == 'eng':
            final_first_part = self.correlation_mil[final_mil] + self.correlation_cen_eng[final_cen]
        else:
            final_first_part = self.correlation_mil[final_mil] + self.correlation_cen_tha[final_cen]

        final_ten = max_ten[0]
        final_yr = max_yr[0]
        if final_ten == '':
            final_ten = str(random.randint(0, 9))
        if final_yr == '':
            final_yr = str(random.randint(0, 9))
        final_year = final_first_part + final_ten + final_yr

        if lang == 'eng':
            if date_type == 'bd' and int(final_year) > 2018 and (max_ten[1] >= float(total_count_year * 0.7) and max_yr[1] >= float(total_count_year * 0.7)):
                final_year = '19' + final_ten + final_yr
                max_mil[1] = 0
                max_cen[1] = 0
        else:
            if date_type == 'bd' and int(final_year) > 2561 and (max_ten[1] >= float(total_count_year * 0.7) and max_yr[1] >= float(total_count_year * 0.7)):
                final_year = '24' + final_ten + final_yr
                max_mil[1] = 0
                max_cen[1] = 0

        final_month = self.clean_date(max_month[0], lang)
        months_without_points = [m.replace('.', '') for m in self.months[lang]]
        if final_month in months_without_points:
            final_month = self.months[lang][months_without_points.index(final_month)]
        if final_month in self.months[lang]:
            month_number = self.months[lang].index(final_month)
            month_days = self.months_days[month_number]
            if int(final_year) % 4 == 0 and month_number == 1:
                month_days += 1
        else:
            month_days = 31
            max_month[1] = 0

        final_day1 = max_day1[0] if max_day1[1] > (total_count_day * 0.3) else ''
        final_day0 = self.correlation_day0[max_day0[0]] if final_day1 != '' else max_day0[0]
        if final_day0 == '' and final_day1 == '':
            final_day = ''
        else:
            if final_day0 != '-' and int(final_day0 + final_day1) > month_days:
                if max_day0[1] > max_day1[1]:
                    final_day1 = self.correlation_day1[final_day1]
                    max_day1[1] = 0
                else:
                    final_day0 = '2'
                    max_day0[1] = 0
                final_day = final_day0 + final_day1
            else:
                final_day = final_day0 + final_day1

        if final_day == '00':
            final_day = '0'
        if final_day == '':
            final_day = '-'
        if final_month == '':
            final_month = '-'
        if final_year == '':
            final_year = '-'
        result = final_day + ' ' + final_month + ' ' + final_year

        log('==========')
        log(' Resulting date from frequencies is \n\t{}'.format(result))
        log('==========')

        freqs = {'day' : max(max_day0[1], max_day1[1]), 'month' : max_month[1], 'year' : max_mil[1] + max_cen[1] + max_ten[1] + max_yr[1]}
        if detected_lifelong:
            result = self.lifelong_text[lang]
            freqs = {'lifelong' : lifelong}

        return result, freqs, total_count
        ######
        # extract text with different modes
        #returned1 = self.extract_date(img, step,lang,model1, ranges, date_type)
        #returned2 = self.extract_date(img, step,lang,model2, ranges, date_type)

        #if date_type == 'ed' and (len(returned1) == 2 or len(returned2) == 2):
        #    return LIFELONG_TEXT, {'prob' : 100}

        #day1, day_freq1, month1, month_freq1, mil1, mil_freq1, cen1, cen_freq1, ten1, ten_freq1, yr1, yr_freq1 = returned1
        #day2, day_freq2, month2, month_freq2, mil2, mil_freq2, cen2, cen_freq2, ten2, ten_freq2, yr2, yr_freq2 = returned2

        #day, day_freq = self.choose_higher(day1, day_freq1, day2, day_freq2)
        #month, month_freq = self.choose_higher(month1, month_freq1, month2, month_freq2)
        #mil, mil_freq = self.choose_higher(mil1, mil_freq1, mil2, mil_freq2)
        #cen, cen_freq  = self.choose_higher(cen1, cen_freq1, cen2, cen_freq2)
        #ten, ten_freq = self.choose_higher(ten1, ten_freq1, ten2, ten_freq2)
        #yr, yr_freq = self.choose_higher(yr1, yr_freq1, yr2, yr_freq2)

        #return day + ' ' + month + ' ' + mil + cen + ten + yr, {'day' : day_freq, 'month' : month_freq, 'year' : mil_freq + cen_freq + ten_freq + yr_freq}

    def similar(self, a, b):
        return SequenceMatcher(None, a, b).ratio()

    def update_freq(self, value, elems, max_elem):
        if value in elems:
            elems[value] += 1
        else:
            elems[value] = 1
        return max_elem if max_elem[1] > elems[value] else [value, elems[value]]

    def extract_date_img(self, img, date_type, lang, models, range_scale):
        text, freqs, total = self.extract_date_modes_balancing(img, lang, models, range_scale, date_type)

        summ = 0
        for part, freq in freqs.items():
            if part == 'year':
                freq = freq / 4.0
            if total != 0:
                percent = 100 * float(freq) / float(total)
            else:
                percent = 0
            summ += percent
        conf = summ / len(freqs.keys())

        if text == self.lifelong_text[lang]:
            return [self.lifelong_text[lang], conf]

        #day, month, year =  self.find_bd_date(text, self.bd_date_ratio_threshold, self.month_ratio_threshold[lang], self.months[lang], lang)
        day, month, year = text.split()

        log('After regex matching the resulting fields are:')
        log('\tDay = {}'.format(day))
        log('\tMonth = {}'.format(month))
        log('\tYear= {}'.format(year))

        if month == '-' and day != '-':
            day = '-'

        return [day, month, year, freqs, conf]

    def get_month_from_day(text, day,eng_months,eng_month_ratio_threshold):
        "Find the month from date position"
        # find date first and end position
        start_index = text.find(day)
        end_index = start_index + len(day)
        # get month
        month = text[(end_index+1):].split(" ")[0]
        return  self.month_matching(eng_months, month,eng_month_ratio_threshold)

    def find_bd_date(self, text, bd_date_ratio_threshold, eng_month_ratio_threshold, eng_months, lang):
        """Find the title and name"""
        words = text.split(" ")
        words = [w for w in words if w != ""]
        # standart situation when there are 3 words
        if len(words) == 3:
            # check if the words Date and Birth exist
            try:
                year = re.match(r'.*([1-3][0-9]{2,3})', text).group(1)
            except:
                try:
                    year = re.match(r'\d+', text).group(1)
                except:
                    year = ''
            try:
                day = re.match(r'([0-9][0-9]{0,1}) ', text).group(1).replace(" ", "")
            except:
                day = "-"
            try:
                month_text = re.sub("[^a-zA-Z]+", "", text) if lang == 'eng' else text
                month = self.month_matching(eng_months, month_text,eng_month_ratio_threshold)
            except:
                month = "-"
        # if there are 5 words
        elif  len(words) == 1:
            year = re.match(r'.*([1-3][0-9]{3})', text).group(1)
            month = "-"
            day = "-"
        # if there are 2 words
        elif  len(words) == 2:
            # extract year
            year = re.match(r'.*([1-3][0-9]{3})', text).group(1)
            # match the date merged with month text
            try:
                day = re.match(r'([0-9][0-9]{0,1})', text).group(1).replace(" ", "")
                # subtract day text
                month_text = re.sub("[^a-zA-Z]+", "", text) if lang == 'eng' else text
                month = self.month_matching(eng_months, month_text,eng_month_ratio_threshold)
            except:
                day = "-"
                month = self.month_matching(eng_months, text,eng_month_ratio_threshold)
        else:
            try:
                day =  re.match(r'([0-9][0-9]{0,1}) ', text).group(1).replace(" ", "")
            except:
                day = "-"
            try:
                month = self.get_month_from_day(text, day,eng_months,eng_month_ratio_threshold)
            except:
                month = "-"
            try:
                year = re.match(r'.*([1-3][0-9]{3})', text).group(1)
            except:
                year = re.match(r'\d+', text).group(1)
        return day, month, year

    def choose_value(self, value1, freq1, value2, freq2):
        if freq1 > freq2:
            return value1
        else:
            return value2

    def choose_month(self, english, thai, english_freqs, thai_freqs):
        if english == '-' and thai != '-' and thai != '':
            return self.months['eng'][self.months['tha'].index(thai)], thai
        if thai == '-' and english != '-' and english != '':
            return english, self.months['tha'][self.months['eng'].index(english)]
        if thai != english and english != '-' and english != '':
            if english_freqs['month'] > thai_freqs['month']:
                return english, self.months['tha'][self.months['eng'].index(english)]
            elif english_freqs['month'] < (thai_freqs['month'] * 0.5):
                return self.months['eng'][self.months['tha'].index(thai)], thai
            else:
                return english, self.months['tha'][self.months['eng'].index(english)]
        return english, thai

    def choose_year(self, english, thai, english_freqs, thai_freqs, date_type='bd'):
        if english == '-':
            return thai, thai
        if thai == '-':
            return english, english
        if int(thai) != (int(english) + 543):
            if FILTERED_YEARS:
                if int(english) > RANGE_YEAR[date_type]['eng'][1] or int(english) < RANGE_YEAR[date_type]['eng'][0]:
                    return str(int(thai) - 543), thai
                elif int(thai) > RANGE_YEAR[date_type]['tha'][1] or int(thai) < RANGE_YEAR[date_type]['tha'][0]:
                    return english, str(int(english) + 543)

            if english_freqs['year'] > thai_freqs['year']:
                return english, str(int(english) + 543)
            elif english_freqs['year'] < thai_freqs['year']:
                return str(int(thai) - 543), thai
            else:
                return english, str(int(english) + 543)
        return english, thai

    def choose_day(self, english, thai, english_freqs, thai_freqs):
        if english == '-':
            if thai == '0':
                return '-', thai
            else:
                return thai, thai
        if thai == '-':
            if english == '0':
                return english, '-'
            else:
                return english, english
        if thai != english:
            if english == '0':
                return english, '-'
            else:
                if english_freqs['day'] > thai_freqs['day']:
                    return english, english
                elif english_freqs['day'] < thai_freqs['day']:
                    return thai, thai
                else:
                    return english, english
        return english, thai

    def run(self, eng_bd_img, eng_ed_img, tha_bd_img, tha_ed_img, tha_models, eng_models, min_scale, max_scale, step_scale):
        range_scale = np.arange(min_scale, max_scale, step_scale)
        eng_bd_d, eng_bd_m, eng_bd_y, eng_bd_freqs, eng_bd_conf = self.extract_date_img(eng_bd_img, 'bd', 'eng', eng_models, range_scale)
        tha_bd_d, tha_bd_m, tha_bd_y, tha_bd_freqs, tha_bd_conf = self.extract_date_img(tha_bd_img, 'bd', 'tha', tha_models, range_scale)

        expiry_eng = self.extract_date_img(eng_ed_img, 'ed', 'eng', eng_models, range_scale)
        expiry_tha = self.extract_date_img(tha_ed_img, 'ed', 'tha', tha_models, range_scale)

        backup_lifelong = None

        if expiry_eng[0] == 'LIFE LONG':
            if ADJUST_LIFELONG_FP and expiry_tha[0] != 'ตลอดชีพ':
                ed_d, ed_m, ed_y, ed_freqs, ed_conf = expiry_tha
                backup_lifelong = {'lang':'tha', 'ed_d':ed_d, 'ed_m':ed_m, 'ed_y':ed_y, 'ed_freqs':ed_freqs, 'ed_conf':ed_conf}
                expiry_tha[1] = expiry_eng[1]
            expiry_tha[0] = 'ตลอดชีพ'
        elif expiry_tha[0] == 'ตลอดชีพ':
            if ADJUST_LIFELONG_FP and expiry_eng[0] != 'LIFE LONG':
                ed_d, ed_m, ed_y, ed_freqs, ed_conf = expiry_eng
                backup_lifelong = {'lang':'eng', 'ed_d':ed_d, 'ed_m':ed_m, 'ed_y':ed_y, 'ed_freqs':ed_freqs, 'ed_conf':ed_conf}
            expiry_eng = ['LIFE LONG', expiry_tha[1]]

        if expiry_eng[0] != 'LIFE LONG':
            eng_ed_d, eng_ed_m, eng_ed_y, eng_ed_freqs, eng_ed_conf = expiry_eng
            tha_ed_d, tha_ed_m, tha_ed_y, tha_ed_freqs, tha_ed_conf = expiry_tha
            if int(eng_ed_y) > RANGE_YEAR['ed']['eng'][1] or int(eng_ed_y) < RANGE_YEAR['ed']['eng'][0]:
                eng_ed_y = str(int(tha_ed_y) - 543)
            elif int(tha_ed_y) > RANGE_YEAR['ed']['tha'][1] or int(eng_ed_y) < RANGE_YEAR['ed']['tha'][0]:
                tha_ed_y = str(int(eng_ed_y) + 543)

        eng_bd_d, tha_bd_d = self.choose_day(eng_bd_d, tha_bd_d, eng_bd_freqs, tha_bd_freqs)
        eng_bd_m, tha_bd_m = self.choose_month(eng_bd_m, tha_bd_m, eng_bd_freqs, tha_bd_freqs)
        eng_bd_y, tha_bd_y = self.choose_year(eng_bd_y, tha_bd_y, eng_bd_freqs, tha_bd_freqs, date_type='bd')

        if expiry_eng[0] != 'LIFE LONG':
            eng_ed_d, tha_ed_d = self.choose_day(eng_ed_d, tha_ed_d, eng_ed_freqs, tha_ed_freqs)
            eng_ed_m, tha_ed_m = self.choose_month(eng_ed_m, tha_ed_m, eng_ed_freqs, tha_ed_freqs)
            eng_ed_y, tha_ed_y = self.choose_year(eng_ed_y, tha_ed_y, eng_ed_freqs, tha_ed_freqs, date_type='ed')

            if ED_TO_BD and (max(eng_ed_conf, tha_ed_conf) > ED_TO_BD_THRES * max(eng_bd_conf, tha_bd_conf) and
                eng_ed_d != '-' and eng_ed_m != '-' and
                not (eng_ed_d == '31' and eng_ed_m == 'Dec.')):
                # ED used to guess BD
                month_number = self.months['eng'].index(eng_ed_m)
                int_day = int(eng_ed_d) + 1
                if int_day > self.months_days[month_number]:
                    int_day = 1
                    month_number += 1
                    if month_number > 11:
                        month_number = 0
                eng_bd_d = str(int_day)
                eng_bd_m = self.months['eng'][month_number]
                tha_bd_d = eng_bd_d
                tha_bd_m = self.months['tha'][month_number]

            else:
                # BD used to guess ED
                if eng_bd_d != '-' and eng_bd_m != '-':
                    int_day = int(eng_bd_d)
                    int_day = int_day - 1 if int_day != 0 else int_day
                    month_number = self.months['eng'].index(eng_bd_m)
                    if int_day == 0:
                        month_number = month_number - 1
                        if FIX_TYPOS and month_number < 0:
                            month_number = 11
                        elif not FIX_TYPOS and month_number == 0:
                            month_number = 11
                        int_day = self.months_days[month_number]
                    eng_ed_d = str(int_day)
                    eng_ed_m = self.months['eng'][month_number]
                    tha_ed_d = eng_ed_d
                    tha_ed_m = self.months['tha'][month_number]
                elif eng_bd_d == '-' and eng_bd_m != '-':
                    month_number = self.months['eng'].index(eng_bd_m) - 1
                    if month_number < 0:
                        month_number = 11
                    int_day = self.months_days[month_number]
                    eng_ed_d = str(int_day)
                    eng_ed_m = self.months['eng'][month_number]
                    tha_ed_d = eng_ed_d
                    tha_ed_m = self.months['tha'][month_number]
                elif eng_bd_d == '-' and eng_bd_m == '-':
                    month_number = 11
                    int_day = 31
                    eng_ed_d = str(int_day)
                    eng_ed_m = self.months['eng'][month_number]
                    tha_ed_d = eng_ed_d
                    tha_ed_m = self.months['tha'][month_number]


            if int(eng_ed_y) > RANGE_YEAR['ed']['eng'][1] and (eng_ed_y[2] == '4' or eng_ed_y[2] == '3'):
                eng_ed_y = str(int(eng_ed_y) - 20)
                tha_ed_y = str(int(tha_ed_y) - 20)
            if int(eng_ed_y) < 2010 and eng_ed_y[2] == '0':
                eng_ed_y = str(int(eng_ed_y) + 20)
                tha_ed_y = str(int(tha_ed_y) + 20)

            if eng_bd_m == '-' and eng_bd_d != '-' and eng_ed_m != '-' and eng_ed_d != '-' and eng_ed_d == '31' and eng_ed_m == 'Dec.':
                eng_bd_d = '-'

            eng_ed_date = eng_ed_d + ' ' + eng_ed_m + ' ' + eng_ed_y
            tha_ed_date = tha_ed_d + ' ' + tha_ed_m + ' ' + tha_ed_y

            if ADJUST_LIFELONG_FN:
                if max(eng_bd_conf, tha_bd_conf) > LIFELONG_FN_THRES and int(eng_bd_y) < UNC_LL_YEAR[0]:
                    eng_ed_date = 'LIFE LONG'
                    tha_ed_date = 'ตลอดชีพ'
                    tha_ed_conf = max(eng_bd_conf, tha_bd_conf)
                    eng_ed_conf = max(eng_bd_conf, tha_bd_conf)

        elif backup_lifelong is not None and int(eng_bd_y) > UNC_LL_YEAR[1] and max(eng_bd_conf, tha_bd_conf) > LIFELONG_FP_THRES * backup_lifelong['ed_conf']:
            if backup_lifelong['lang'] == 'eng':
                eng_ed_d, eng_ed_m, eng_ed_y = backup_lifelong['ed_d'], backup_lifelong['ed_m'], backup_lifelong['ed_y']
                tha_ed_d = backup_lifelong['ed_d']
                tha_ed_m = self.months['tha'][self.months['eng'].index(eng_ed_m)]
                tha_ed_y = str(543 + int(eng_ed_y))
                tha_ed_date = tha_ed_d + ' ' + tha_ed_m + ' ' + tha_ed_y
                eng_ed_date = eng_ed_d + ' ' + eng_ed_m + ' ' + eng_ed_y
                tha_ed_conf = backup_lifelong['ed_conf']
                eng_ed_conf = backup_lifelong['ed_conf']
            else:
                tha_ed_d, tha_ed_m, tha_ed_y = backup_lifelong['ed_d'], backup_lifelong['ed_m'], backup_lifelong['ed_y']
                eng_ed_d = backup_lifelong['ed_d']
                eng_ed_m = self.months['eng'][self.months['tha'].index(tha_ed_m)]
                eng_ed_y = str(int(tha_ed_y) - 543)
                tha_ed_date = tha_ed_d + ' ' + tha_ed_m + ' ' + tha_ed_y
                eng_ed_date = eng_ed_d + ' ' + eng_ed_m + ' ' + eng_ed_y
                tha_ed_conf = backup_lifelong['ed_conf']
                eng_ed_conf = backup_lifelong['ed_conf']

        else:
            eng_ed_date = 'LIFE LONG'
            tha_ed_date = 'ตลอดชีพ'
            if eng_bd_d != '0':
                if eng_bd_d == tha_bd_d:
                    eng_bd_d = '0'
                else:
                    eng_bd_d = '-'
            if eng_bd_m == '-' and eng_bd_d != '-':
                eng_bd_d = '-'
            tha_bd_d = '-'
            eng_ed_conf = expiry_eng[1]
            tha_ed_conf = expiry_tha[1]

        eng_bd_date = eng_bd_d + ' ' + eng_bd_m + ' ' + eng_bd_y
        tha_bd_date = tha_bd_d + ' ' + tha_bd_m + ' ' + tha_bd_y

        log('==========')
        log('Thai BD date: {}'.format(tha_bd_date))
        log('Eng. BD date: {}'.format(eng_bd_date))
        log('Thai ED date: {}'.format(tha_ed_date))
        log('Eng. ED date: {}'.format(eng_ed_date))
        log('Thai BD conf: {}'.format(tha_bd_conf))
        log('Eng. BD conf: {}'.format(eng_bd_conf))
        log('Thai ED conf: {}'.format(tha_ed_conf))
        log('Eng. ED conf: {}'.format(eng_ed_conf))
        log('==========')

        return collections.OrderedDict([
            ('tha_bd_date', tha_bd_date),
            ('eng_bd_date', eng_bd_date),
            ('tha_ed_date', tha_ed_date),
            ('eng_ed_date', eng_ed_date),
            ('tha_bd_date_conf', tha_bd_conf),
            ('eng_bd_date_conf', eng_bd_conf),
            ('tha_ed_date_conf', tha_ed_conf),
            ('eng_ed_date_conf', eng_ed_conf)
        ])

de = DateExtractor()
eng_models = ['--psm 12 --oem 1']
tha_models = ['--psm 4 --oem 1']
min_scale = 1
max_scale = 10
step_scale = 0.2
def extract_dates(field_box_imgs_denoised, tha_models=tha_models, eng_models=eng_models, min_scale=min_scale, max_scale=max_scale, step_scale=step_scale):
    if field_box_imgs_denoised == None:
        return collections.OrderedDict([
            ('tha_bd_date', ''),
            ('eng_bd_date', ''),
            ('tha_ed_date', ''),
            ('eng_ed_date', ''),
            ('tha_bd_date_conf', 0),
            ('eng_bd_date_conf', 0),
            ('tha_ed_date_conf', 0),
            ('eng_ed_date_conf', 0)
        ])
    eng_bd_img = field_box_imgs_denoised['bd_eng']
    eng_ed_img = field_box_imgs_denoised['ed_eng']
    tha_bd_img = field_box_imgs_denoised['bd_tha']
    tha_ed_img = field_box_imgs_denoised['ed_tha']
    return de.run(eng_bd_img, eng_ed_img, tha_bd_img, tha_ed_img, tha_models, eng_models, min_scale, max_scale, step_scale)

def get_args():
    parser = argparse.ArgumentParser(description='Extract dates from images that contain them.')
    parser.add_argument('eng_bd_path', help='path to the image from which to extract')
    parser.add_argument('eng_ed_path', help='path to the image from which to extract')
    parser.add_argument('tha_bd_path', help='path to the image from which to extract')
    parser.add_argument('tha_ed_path', help='path to the image from which to extract')
    return parser.parse_args()

def main():
    args = get_args()
    de = DateExtractor()
    img1 = Image.open(args.eng_bd_path)
    img2 = Image.open(args.eng_ed_path)
    img3 = Image.open(args.tha_bd_path)
    img4 = Image.open(args.tha_ed_path)
    result = de.run(img1, img2, img3, img4, tha_models, eng_models, min_scale, max_scale, step_scale)

if __name__ == '__main__':
    main()

