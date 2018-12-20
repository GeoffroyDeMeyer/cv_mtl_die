import numpy as np
import pandas as pd
import re
import pytesseract
import itertools
from collections import OrderedDict, Counter
import ast
import random
from googletrans import Translator

# to reproduce results
SEED = 7
random.seed(SEED)

try:
    import address_extraction.utils as utils
except ImportError:
    import utils

RESOURCE_PATH   = 'address_extraction/resources/'
TESS_CONF3       = '--oem 1 --psm 3 -c preserve_interword_spaces=1'
TESS_CONF4       = '--oem 1 --psm 4 -c preserve_interword_spaces=1'
TESS_CONF6       = '--oem 1 --psm 6 -c preserve_interword_spaces=1'
TESS_CONF8       = '--oem 1 --psm 8 -c preserve_interword_spaces=1'
TESS_CONF12       = '--oem 1 --psm 12 -c preserve_interword_spaces=1'
TESS_CONF13       = '--oem 1 --psm 13 -c preserve_interword_spaces=1'

NOT_FOUND = (-1, '')
UNKNOWN_BANGKOK_TYPE = -1

COMP_SIM_THESHOLD = 0.8

class AddressExtractor:
    def __init__(self):
        self.tesseract_confs = [TESS_CONF6, TESS_CONF8]#[TESS_CONF6, TESS_CONF8, TESS_CONF12, TESS_CONF13]#[TESS_CONF3, TESS_CONF4, TESS_CONF6, TESS_CONF8, TESS_CONF12, TESS_CONF13]

        self.symbol_dict = pd.read_csv(RESOURCE_PATH + 'symbols.csv').set_index('name').squeeze().to_dict()
        self.component_dict = pd.read_csv(RESOURCE_PATH + 'components.csv').set_index('name').squeeze().to_dict()
        self.changwats = pd.read_csv(RESOURCE_PATH + 'changwats.csv').changwat.values
        self.amphoes = pd.read_csv(RESOURCE_PATH + 'amphoes.csv').amphoe.values
        self.tambon2amphoe2changwat = pd.read_csv(RESOURCE_PATH + 'tambon2amphoe2changwat.csv')
        self.tambons = self.tambon2amphoe2changwat.tambon.values
        self.khets = pd.read_csv(RESOURCE_PATH + 'khet2khwaengs.csv').khet.values
        self.khet2khwaengs = pd.read_csv(RESOURCE_PATH + 'khet2khwaengs.csv')
        self.khet2khwaengs.khwaengs = self.khet2khwaengs.khwaengs.map(lambda x: ast.literal_eval(x))
        self.khwaengs = [item for sublist in self.khet2khwaengs.khwaengs.values for item in sublist]

        self.khet2khwaengs_dict = self.khet2khwaengs.set_index('khet').squeeze().to_dict()
        self.khwaeng2khet_dict = utils.invert_dict(self.khet2khwaengs_dict)

        self.translator = Translator()

    def run(self, fields_address, max_scale, scale_step):

        if fields_address is None:
            return OrderedDict([
            ('address_l1', ""),
            ('address_l2', ""),
            ('address_l1_conf', 0.0),
            ('address_l2_conf', 0.0),
        ])

        img_l1 = fields_address["address_l1"]
        img_l2 = fields_address["address_l2"]

        self.scaling_range = np.arange(1, max_scale, scale_step)
        
        l1_values = self.extract_values(img_l1)
        l2_values = self.extract_values(img_l2)

        index, c_i, address, l2_fields = self.build_frequency_index(l1_values, l2_values)

        # print(index)

        self.translate(index, 'something')
        self.translate(index, 'street')

        if AddressExtractor.is_bangkok_address(index):
            address_l1, address_l2, address_l1_conf, address_l2_conf = self.process_bangkok(index, c_i, address, l2_fields)
        else:
            address_l1, address_l2, address_l1_conf, address_l2_conf = self.process_non_bangkok(index, c_i, address, l2_fields)


        return OrderedDict([
            ('address_l1', address_l1),
            ('address_l2', address_l2),
            ('address_l1_conf', address_l1_conf),
            ('address_l2_conf', address_l2_conf),
        ])

    def build_frequency_index(self, l1_values, l2_values):
        '''
        Return an indexed address with all fields and positions 'field': (idx, value)
        The key 'original' gives the most frequent address computed word wise
        '''
        l1_values_or = l1_values
        l2_values_or = l2_values

        l1_values = list(map(self.clean_address_line, l1_values))
        l2_values = list(map(self.clean_address_line, l2_values))

        l1_values, length_l1 = utils.most_frequent_length(l1_values)
        l2_values, length_l2 = utils.most_frequent_length(l2_values)

        l1_values, count_figures_l1 = utils.most_frequent_amount_figures(l1_values)
        l2_values, count_figures_l2 = utils.most_frequent_amount_figures(l2_values)

        all_addresses = list(map(lambda t: ' '.join(t), itertools.product(l1_values, l2_values)))
        all_indices = list(map(self.extract_index, all_addresses))
        index, c_i = utils.most_frequent_by_key(all_indices)

        l1_values = utils.most_frequent_words(l1_values)
        l2_values = utils.most_frequent_words(l2_values)

        # print()
        # print(l1_values, ' -- ', l2_values, ' (tesseract)')
        # print()
        # print(index)

        l2_fields = self.extract_index(l2_values).keys()

        # print()
        # print('keys for l2: ', l2_fields)

        address = ' '.join([l1_values, l2_values])

        return index, c_i, address, l2_fields

    def brute_force_empty_fields(self, index, address_words):
        # print(5*'*')

        if AddressExtractor.is_bangkok_address(index):
            field2dict = {'khwaeng': self.khwaengs, 'khet': self.khets}
        else:
            field2dict = {'tambon': self.tambons, 'amphoe': self.amphoes, 'changwat': self.changwats}
        
        brute_force_fields = []
        for k,v in field2dict.items():
            if k not in index.keys():
                brute_force_fields.append((k, v))

        found_indices = []
        for k,v in index.items():
            if v != NOT_FOUND:
                found_indices.append(v[0])

        reduced_address_words = []
        for i, word in enumerate(address_words):
            if len(word) > 2 and i not in found_indices:
                reduced_address_words.append((i, word))

        # print(reduced_address_words)
        if len(brute_force_fields) != 0:
            for i, word in reduced_address_words:

                best_value, best_score, best_field = '', 0, None
                for field, field_dict in brute_force_fields:
                    
                    curr_value, curr_score = utils.word_matching(field2dict[field], word)
                    if curr_score > best_score:
                        best_score = curr_score
                        best_value = curr_value
                        best_field = field

                index[best_field] = (i, word)
        
        # print([k for k,v in brute_force_fields])

    def extract_index(self, address):
        words = address.split()
        index = {}

        # get component in index
        index = self.search_symbol(index, address)

        # get symbol in index
        index = self.search_component(index, address)

        for i, word in enumerate(words):
            if utils.is_figure(word):
                # to avoid 0 in output
                if word == "0":
                    word = "8"

                if 'number1' in index:
                    index['number2'] = (i, word)
                else:
                    index['number1'] = (i, word)

        return index

    def search_symbol(self, index, address, min_size=2):

        for k, v in self.symbol_dict.items():
            take_next = False
            for i, word in enumerate(address.split()):
                if take_next:
                    # we take the next value only if big enough
                    if len(word) >= min_size:
                        index[k] = (i, word)
                        break
                    else:
                        take_next = False

                if word.startswith(v):

                    if word == v:
                        # we found the the symbol but only the symbol
                        # we take the next value
                        take_next = True
                    elif len(word) >= (min_size + 1):
                        # the symbol is concat with the actual value
                        # ask for min_size + 1
                        index[k] = (i, word[1:])

        return index

    def search_component(self, index, address, min_size=3):

        for k,v in self.component_dict.items():
            # only process if not yet found
            if not k in index:

                if k != 'bangkok' and k != "city_num":
                    for i, word in enumerate(address.split()):

                        prefix_length = len(v)
                        prefix = word[:prefix_length]

                        # we need the same size here
                        if len(prefix) == prefix_length:
                            if k == "khet":
                                if prefix[1] == "ปิ" or prefix[1] == "ซ":
                                    prefix = prefix[0] + "ข" + prefix[2]
                                if prefix[2] == "ดี" or prefix[2] == "ด":
                                    prefix = prefix[:2] + "ต"

                            score = 0
                            for char_res, char_true in list(zip(prefix, v)):
                                score += 1 if char_res == char_true else 0

                            if score/prefix_length >= 0.49 and (len(word) - prefix_length) >= min_size:
                                index[k] = (i, word[prefix_length:])
                                break
                
                else:
                    for i, word in enumerate(address.split()):

                        if k == "city_num":
                            word = word.replace("มู","มู่")

                        if utils.similar(word, v) >= 0.49:
                            index[k] = (i, v)

        return index

    def process_non_bangkok(self, index, c_i, address, l2_fields):
        
        best_score_c = self.check_similarity(index, 'changwat', self.changwats)
        best_score_a = self.check_similarity(index, 'amphoe', self.amphoes)
        best_score_t = self.check_similarity(index, 'tambon', self.tambons)

        # print('-before', index)
        self.brute_force_empty_fields(index, address)
        self.check_correctness(index, is_bangkok=False, bs_tambon=best_score_t, bs_amphoe=best_score_a, bs_changwat=best_score_c)
        # print('-after', index)

        default_l1_fields = ['number1', 'city_num', 'number2', 'street', 'something', 'tambon', 'amphoe']
        default_l2_fields = ['changwat']

        l1_fields, l2_fields = self.update_l1_l2_fields_non_bangkok(default_l1_fields, default_l2_fields, l2_fields)

        address_l1 = self.format_address(index, l1_fields)
        address_l2 = self.format_address(index, l2_fields)

        address_l1_conf = self.get_c_i(index, c_i, default_l1_fields)
        address_l2_conf = self.get_c_i(index, c_i, default_l2_fields)

        return address_l1, address_l2, address_l1_conf, address_l2_conf

    def process_bangkok(self, index, c_i, address, l2_fields):

        best_score_ke = self.check_similarity(index, 'khet', self.khets)
        best_score_ka = self.check_similarity(index, 'khwaeng', self.khwaengs)

        # print('-before', index)
        self.brute_force_empty_fields(index, address)
        self.check_correctness(index, is_bangkok=True, bs_khet=best_score_ke, bs_khwaeng=best_score_ka)
        # print('-after', index)

        default_l1_fields = ['number1', 'something', 'number2', 'street', 'khwaeng']
        default_l2_fields = ['khet', 'bangkok']

        l1_fields, l2_fields = self.update_l1_l2_fields_bangkok(default_l1_fields, default_l2_fields, l2_fields)

        address_l1 = self.format_address(index, l1_fields)
        address_l2 = self.format_address(index, l2_fields)

        address_l1_conf = self.get_c_i(index, c_i, default_l1_fields)
        address_l2_conf = self.get_c_i(index, c_i, default_l2_fields)

        return address_l1, address_l2, address_l1_conf, address_l2_conf

    def update_l1_l2_fields_non_bangkok(self, default_l1_fields, default_l2_fields, input_l2_fields):
        
        l1_field = []
        l2_field = []

        if 'tambon' in input_l2_fields:
            l1_field = ['number1', 'city_num', 'number2', 'street', 'something']
            l2_field = ['tambon', 'amphoe', 'changwat']
            return l1_field, l2_field
        elif 'amphoe' in input_l2_fields:
            l1_field = ['number1', 'city_num', 'number2', 'street', 'something', 'tambon']
            l2_field = ['amphoe', 'changwat']
            return l1_field, l2_field
        else:
            l1_field = ['number1', 'city_num', 'number2', 'street', 'something', 'tambon', 'amphoe']
            l2_field = ['changwat']
            return default_l1_fields, default_l2_fields

    def update_l1_l2_fields_bangkok(self, default_l1_fields, default_l2_fields, input_l2_fields):
        
        l1_field = []
        l2_field = []

        if 'khwaeng' in input_l2_fields:
            l1_field = ['number1', 'something', 'number2', 'street']
            l2_field = ['khwaeng', 'khet', 'bangkok']
            return l1_field, l2_field
        elif 'khet' in input_l2_fields:
            return default_l1_fields, default_l2_fields
        else:
            l1_field = ['number1', 'something', 'number2', 'street', 'khwaeng']
            l2_field = ['bangkok']
            return l1_field, l2_field

    def format_address(self, index, fields):
        res = []
        for f in fields:
            if f in index:
                value = index[f][1]
                if value != '':
                    if f in self.symbol_dict.keys():
                        res.append(self.symbol_dict[f] + '.' + value)
                    elif f in self.component_dict.keys():
                        if f == "bangkok" or f == "city_num":
                            res.append(self.component_dict[f])
                        else:
                            res.append(self.component_dict[f] + value)
                    else:
                        res.append(utils.clean_number(value))
            elif f == "changwat":
                res.append(self.symbol_dict[f] + ".ชลบุรี")

        return ' '.join(res)

    def get_c_i(self, index, c_i, fields):
        res = []
        for field in fields:
            if field in index.keys() and index[field][1] != '':
                if field in c_i.keys():
                    res.append(c_i[field])
                else:
                    res.append(0)

        return np.mean(res)

    def check_similarity(self, index, field, field_db):
        if field in index:
            best_value, best_score = utils.word_matching(field_db, index[field][1])
            index[field] = (index[field][0], best_value)
            return best_score
        else:
            return 0.0

    def random_choice(self, l):
        return "" if len(l) == 0 else random.choice(l)

    def google_tha_to_tha(self, word):
        tr = self.translator.translate(word, src='th').extra_data['possible-mistakes']
        if not tr is None and len(tr) > 1:
            # print("Extra data", tr)
            return tr[1]
        else:
            return word

    def translate(self, index, field):
        if field in index:
            # print("Before translation ",field, index[field])
            index[field] = (index[field][0], self.google_tha_to_tha(index[field][1]))
            # print("After translation  ",field, index[field])
        return index

    def check_correctness(self, index, is_bangkok, bs_khet=0.0, bs_khwaeng=0.0, bs_tambon=0.0, bs_amphoe=0.0, bs_changwat=0.0):
        default_index = -2
        if is_bangkok:

            if 'khwaeng' not in index.keys() and 'khet' in index.keys():
                # print(1)
                (_, khet) = index['khet']
                khwaeng_candidates = self.khet2khwaengs_dict[khet]
                index['khwaeng'] = (default_index, self.random_choice(khwaeng_candidates))
            
            elif 'khwaeng' in index.keys() and 'khet' not in index.keys():
                # print(2)
                (_, khwaeng) = index['khwaeng']
                khet_candidates = self.khwaeng2khet_dict[khwaeng]
                index['khet'] = (default_index, self.random_choice(khet_candidates))
            
            elif 'khwaeng' in index.keys() and 'khet' in index.keys():
                # print(3)
                (_, khwaeng), (_, khet) = index['khwaeng'], index['khet']
                if bs_khwaeng < bs_khet:
                    khwaeng_candidates = self.khet2khwaengs_dict[khet]
                    index['khwaeng'] = (default_index, utils.word_matching(khwaeng_candidates, khwaeng)[0])
                else:
                    khet_candidates = self.khwaeng2khet_dict[khwaeng]
                    index['khet'] = (default_index, utils.word_matching(khet_candidates, khet)[0])

        else:
            df = self.tambon2amphoe2changwat

            if 'tambon' not in index.keys() and 'amphoe' in index.keys() and 'changwat' in index.keys():
                # print(4)
                (_, amphoe), (_, changwat) = index['amphoe'], index['changwat']
                tambon_candidates = df.loc[(df['amphoe'] == amphoe) & (df['changwat'] == changwat)].tambon.values
                index["tambon"] = (default_index, self.random_choice(tambon_candidates))
            
            elif 'tambon' in index.keys() and 'amphoe' not in index.keys() and 'changwat' in index.keys():
                # print(5)
                (_, tambon), (_, changwat) = index['tambon'], index['changwat']
                amphoe_candidates = df.loc[(df['tambon'] == tambon) & (df['changwat'] == changwat)].amphoe.values
                index["amphoe"] = (default_index, self.random_choice(amphoe_candidates))

            elif 'tambon' in index.keys() and 'amphoe' in index.keys() and 'changwat' not in index.keys():
                # print(6)
                (_, tambon), (_, amphoe) = index['tambon'], index['amphoe']
                changwat_candidates = df.loc[(df['tambon'] == tambon) & (df['amphoe'] == amphoe)].changwat.values
                index["changwat"] = (default_index, self.random_choice(changwat_candidates))
            
            elif 'tambon' in index.keys() and 'amphoe' in index.keys() and 'changwat' in index.keys():
                (_, tambon), (_, amphoe), (_, changwat) = index['tambon'], index['amphoe'], index['changwat']
                if bs_tambon < bs_changwat and bs_tambon < bs_amphoe:
                    # print(7)
                    tambon_candidates = df.loc[(df['amphoe'] == amphoe) & (df['changwat'] == changwat)].tambon.values
                    index['tambon'] = (-2, "") if len(tambon_candidates) == 0 else (default_index, tambon) if tambon in tambon_candidates else (default_index, utils.word_matching(tambon_candidates, tambon)[0])

                elif bs_amphoe < bs_tambon and bs_amphoe < bs_changwat:
                    # print(8)
                    amphoe_candidates = df.loc[(df['tambon'] == tambon) & (df['changwat'] == changwat)].amphoe.values
                    index['amphoe'] = (-2, "") if len(amphoe_candidates) == 0 else (default_index, amphoe) if amphoe in amphoe_candidates else (default_index, utils.word_matching(amphoe_candidates, amphoe)[0])

                else:
                    # print(9)
                    changwat_candidates = df.loc[(df['tambon'] == tambon) & (df['amphoe'] == amphoe)].changwat.values
                    index['changwat'] = (-2, "") if len(changwat_candidates) == 0 else (default_index, changwat) if changwat in changwat_candidates else (default_index, utils.word_matching(changwat_candidates, changwat)[0])

        return index

    def apply_tesseract(self, scale, config, img):
        return pytesseract.image_to_string(utils.resize(img, scale), 'tha', config=config)

    def extract_values(self, img):
        return [self.apply_tesseract(scale, conf, img) for conf in self.tesseract_confs for scale in self.scaling_range]

    @staticmethod
    def is_bangkok_address(index):
        return "bangkok" in index or ("khet" in index and "khwaeng" in index)

    @staticmethod
    def bangkok_type(index):
        if 'something' in index and "number_2" in index:
            return 1
        elif 'street' in index and "number_2" in index:
            return 2
        
        return UNKNOWN_BANGKOK_TYPE

    def isolate_number(self, line):
        res = ""
        prev = ""
        prev2 = ""
        digit = set(['0','1','2','3','4','5','6','7','8','9'])
        for c in line:
            if c in digit:
                if prev in digit or (prev == '/' and prev2 in digit) or prev == " ":
                    res += c
                else:
                    res += " " + c
            else:
                if prev in digit and c != " " and c != "/":
                    res += " "
                res += c
            prev2 = prev
            prev = c
        return res.strip()

    def clean_address_line(self, line):     
        blacklist = '!-\_"\'“[];\{\}€”<>+=*:|`@?#°()'
        line = line.replace('\n', ' ')
        line = line.replace(',', '.')
        line = line.replace('.', ' ')
        line = self.isolate_number(line)

        # TODO im 434
        # isolate first relevant char => too dangerous 
        #line = " " + line
        #line = line.replace(" ซ", " ซ ").replace(" ถ", " ถ ").replace(" ต", " ต ").replace(" อ", " อ ").replace(" จ", " จ ")

        # replace almost similar character
        # tambon
        line = line.replace(" ด ", " ต ")
        # amphoe
        line = line.replace(" ล ", " อ ").replace(" อั",  " อ ").replace(" อํ", " อ ").replace(" ล", " อ ")
        # something
        line = line.replace(" ช ", " ซ ")

        chars = list(map(lambda c: c if c not in blacklist else ' ', line))

        return (''.join(chars)).strip()

