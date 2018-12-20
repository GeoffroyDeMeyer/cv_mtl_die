import pytesseract
from PIL import Image

import numpy as np
import pandas as pd
from collections import Counter, OrderedDict

try:
    import address_extraction.address_utils as address_utils
    import address_extraction.parse_address_utils as parse_address_utils
    import address_extraction.address_api as address_api 
except ModuleNotFoundError:
    import address_utils, parse_address_utils, address_api

RESOURCE_PATH = 'address_extraction/resources/'
CONFIG  = "--oem 1 --psm 12 -c preserve_interword_spaces=1"
STEP = 0.2
SCALES = np.arange(1, 5, 0.5)
LANG = 'tha'

DEBUG = False
#DEBUG = True 

def log(*args):
    if DEBUG:
        print(''.join(map(str,args)))

def find_index(l, lookup_symbol):
    """Return the index of the lookup_symbol from the list l (-1 if it is not found)."""
    index = -1
    for i, e in enumerate(l):
        if e == lookup_symbol:
            index = i
    return index

symbol_dict = {
    'address': 'ที่อยู่',
    'street': 'ถ',
    'township': 'ต',
    'group': 'หมู่ที่',
    'district': 'อ',
    'district_bangkok': 'เขต',
    'province': 'จ',
    'city': 'เมือง'
}

class AddressExtractor:

    def __init__(self, resource_path=RESOURCE_PATH, config=CONFIG, step=STEP, scales=SCALES):
        self.tags = pd.read_csv(resource_path+"tags.csv").set_index('name').squeeze().to_dict()
        self.provinces = pd.read_csv(resource_path+"provinces.csv").province.values
        self.districts = pd.read_csv(resource_path+"districts.csv").district.values
        self.district2province = pd.read_csv(resource_path+"district2province.csv").set_index('district').squeeze().to_dict()
        self.config = config
        self.step = step
        self.scales = scales

    def run(self, address_l1_img, address_l2_img):
        # Extract and combine lines
        log('\nExtraction for address l1:')
        text_address_l1 = self.extract_string(address_l1_img)
        log('\nExtraction for address l2:')
        text_address_l2 = self.extract_string(address_l2_img)

        return self.process_tesseract_output(text_address_l1, text_address_l2)

    def extract_string(self, img):
        # get image size
        w, h = img.size

        # define variables
        freqs = {}
        max_freq = ("", 0)

        # iterate through size
        for scale in self.scales:
            # size values
            wb = int(w * scale)
            hb = int(h * scale)

            # text extraction
            txt = pytesseract.image_to_string(img.resize((wb, hb)),
                LANG, config=self.config)

            cleaned_txt = address_utils.clean_text(txt)
            log(cleaned_txt)

            # find the most frequent result
            if cleaned_txt != "":
                if cleaned_txt in freqs:
                    freqs[cleaned_txt] += 1
                else:
                    freqs[cleaned_txt] = 1

                if freqs[cleaned_txt] > max_freq[1]:
                    max_freq = (cleaned_txt, freqs[cleaned_txt])

        result = max_freq[0]

        log('\n' + 5*'*')
        log('Result ({}/{}): {}'.format(max_freq[1], len(self.scales), result))

        return result

    def process_tesseract_output(self, text_address_l1, text_address_l2):
        """Process the output of Tesseract."""

        # combine all address lines
        text_address = text_address_l1 + " " + text_address_l2

        # get all the information we can retrieve (street, province, district, ...)
        parsed_address = self.parse_address(text_address)

        # split back all the information into two lines
        # splitted_l1, splitted_l2 = separate_lines(
        #     parsed_address,
        #     text_address_l1,
        #     text_address_l2
        #     )

        address_l1 = ''

        # by default and for consistency I will only and always put the province on the
        # second line, we still need to determine how we will compute acc with labels

        address_l2 = '. '.join([symbol_dict['province'], parsed_address['province'][0]])

        return OrderedDict([
            ('address_l1', address_l1),
            ('address_l2', address_l2)
        ])

    def parse_address(self, text):
        """Process address text extracted from Tesseract."""

        # init output
        empty_text = ''
        default_index = -1
        res = {
            'number': (empty_text, default_index),
            'address': (empty_text, default_index),
            'street': (empty_text, default_index),
            'township': (empty_text, default_index),
            'group': (empty_text, default_index),
            'district': (empty_text, default_index),
            'district_bangkok': (empty_text, default_index),
            'province': (empty_text, default_index),
            'city': (empty_text, default_index)
        }

        # first thing to do is check if we can found any symbol that is redundant in the addresses
        words = [w for w in text.split(" ") if w != ""]
        log('\nwords:', words)

        for k,v in symbol_dict.items():
            current_index = find_index(words, lookup_symbol=v)

            # if we find a symbol then we just read the word just after it
            if current_index != -1:
                log('Found {} symbol in address.'.format(k))
                res[k] = (res[k][0], current_index+1)
                
                # if we have a database with all the possibilities corresponding the the symbol
                # then we just take the candidate with highest similarity score
                if k == 'district':
                    best_matching_field = word_matching(self.districts, words[res[k][1]])
                    log('Best match for district: ', best_matching_field)
                    res[k] = (best_matching_field[0], res[k][1])
                elif k == 'province':
                    best_matching_field = word_matching(self.districts, words[res[k][1]])
                    log('Best match for province: ', best_matching_field)
                    res[k] = (best_matching_field[0], res[k][1])

        # if we now the district but the province is empty, then we can infer the province
        if res['district'][1] != -1 and res['province'][1] == -1:
            res['province'] = (self.district2province[res['district'][0]], -2)

        # rebuilt output
        # output = []
        # for k,v in res.items():
        #     output.append(v[0])
        # output = ' '.join(output)
        
        return res

        # index_disctrict_symbol = find_index(words, lookup_symbol='อ')
        # print('index district symbol: ', index_disctrict_symbol)

        # if index_disctrict_symbol is not -1:
        #     # the district word is right after the district symbol
        #     district_word = words[index_disctrict_symbol + 1]
        #     print('District is represented by the following word (with high prob): ', district_word)
        #     check_disctrict_word(district_word, self.districts)

        # index_province_symbol = find_index(words, lookup_symbol='จ')
        # print('index province symbol: ', index_province_symbol)

        # if index_province_symbol is not -1:
        #     # the district word is right after the district symbol
        #     district_word = words[index_province_symbol + 1]
        #     print('Province is represented by the following word (with high prob): ', district_word)
        #     check_disctrict_word(district_word, self.districts)

        # found_district_province = find_district_province(words,
        #     self.districts, self.provinces, nb_results=5)

        # print(found_district_province)

        # if found_district_province is not None:
        #     most_likely = select_most_likely_district_province(
        #         found_district_province,
        #         self.province_district)

        #     print("Found province-district:", most_likely)

        #     if most_likely is not None:
        #         district = most_likely['district']
        #         province = most_likely['province']
        #         length   = len(words) - most_likely['parsed']
        #         return autocorrect_address(" ".join(words[:length]) +
        #         " {d} {p}".format(d=district, p=province))

        #     else:
        #         return autocorrect_address(" ".join(words))

        # else:
        #     return autocorrect_address(" ".join(words))

ae = AddressExtractor()
def extract_addresses(address_l1_img, address_l2_img):
    try:
        return ae.run(address_l1_img, address_l2_img)
    except TypeError:
        return OrderedDict([
            ('address_l1', "TypeError"),
            ('address_l2', "TypeError")
        ])