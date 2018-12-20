import pytesseract, unidecode
from PIL import Image
from googletrans import Translator
from pythainlp.romanization import romanization

import numpy as np
import pandas as pd
from collections import OrderedDict
import unicodedata

try:
    import name_extraction.utils as utils
except ModuleNotFoundError:
    import utils

RESOURCE_PATH           = 'name_extraction/resources/'
TITLE_MIN_SIM           = 0.8
DEFAULT_TITLE           = 'Mr.'
INCLUDED_TITLE_RATIO    = 0.3
THA_MODEL_REQUIRED_FLAG = '-c preserve_interword_spaces=1'
GOOGLE_ENG_TO_THA       = True
EMPTY_FIELD             = ('', 0)

LETTER_CORRECTION = {
    't': 'i',
    'l': 'i',
    's': 'a'
}
CONSONNANTS_EXCPETIONS = ['mrs', 'phr', 'ngm', 'nth', 'stl', 'ngs', 'wph']

HARD_CORRECTION = {
    'Ne': 'Na'
}

class NameExtractor:
    def __init__(self):
        self.title_df = pd.read_csv(RESOURCE_PATH + 'titles.csv')
        self.fistname_df = pd.read_csv(RESOURCE_PATH + 'firstnames.csv')
        self.lastname_df = pd.read_csv(RESOURCE_PATH + 'lastnames.csv')

        self.translator = Translator()

    def run(self, thai_img, eng_fn_img, eng_ln_img, min_scale, max_scale, step_scale, tha_models, eng_models):
        self.scaling_range = np.arange(min_scale, max_scale, step_scale)
        self.eng_configs = eng_models
        self.tha_configs = NameExtractor.validate_tha_models(tha_models)

        (eng_title, eng_title_conf), (eng_fn, eng_fn_conf) = self.eng_extract_firstname(eng_fn_img)
        eng_ln, eng_ln_conf = self.eng_extract_lastname(eng_ln_img)
        eng_fn_len, eng_ln_len = len(eng_fn.split()), len(eng_ln.split())

        (tha_title, tha_title_conf), (tha_fn, tha_fn_conf), (tha_ln, tha_ln_conf) = self.thai_extract(thai_img, eng_fn_len, eng_ln_len)
        if len(tha_fn) > 0:
            tha_fn = NameExtractor.rmdiacritics(tha_fn[0]) + tha_fn[1:]

        tha_fn, tha_ln = self.balance_thai_name(tha_fn, tha_ln)
        tha_fn = self.google_tha_to_tha(tha_fn)
        tha_ln = self.google_tha_to_tha(tha_ln)
        
        result = OrderedDict([
            ('tha_title',           tha_title),
            ('tha_firstname',       tha_fn.replace('.', '')),
            ('tha_lastname',        tha_ln),
            ('eng_title',           eng_title),
            ('eng_firstname',       eng_fn.title()),
            ('eng_lastname',        eng_ln.title()),
            ('tha_title_conf',      tha_title_conf * 100),
            ('tha_firstname_conf',  tha_fn_conf * 100),
            ('tha_lastname_conf',   tha_ln_conf * 100),
            ('eng_title_conf',      eng_title_conf * 100),
            ('eng_firstname_conf',  eng_fn_conf * 100),
            ('eng_lastname_conf',   eng_ln_conf * 100)
        ])

        result = self.translate_missing(result)
        result = self.fill_by_default(result)
        result = self.match_titles(result)
        return result

    def thai_extract(self, thai_img, eng_fn_len, eng_ln_len):
        values, length = self.thai_get_clean_values(thai_img)

        if length == 0:
            # Bad OCR
            return (EMPTY_FIELD, EMPTY_FIELD, EMPTY_FIELD)
        elif length == 1:
            return self.thai_len_1(values)
        elif length == 2:
            return self.thai_len_2(values, eng_fn_len)
        else:
            return self.thai_len_3_more(values, eng_fn_len, eng_ln_len, length)

    def eng_extract_lastname(self, eng_fn_img):
        values, length = self.eng_get_clean_correct_values(eng_fn_img)

        if length == 0:
            return EMPTY_FIELD
        elif length == 1:
            name, ratio = utils.most_frequent(values, with_ratio=True)
        else:
            name, ratio = utils.most_frequent_by_word(values)

        name = ' '.join(list(filter(lambda w: len(w) > 2 or w == 'Na', name.split())))
        return name, ratio

    def eng_extract_firstname(self, eng_ln_img):
        values, length = self.eng_get_clean_correct_values(eng_ln_img)

        if length == 0:
            # Bad OCR
            return (EMPTY_FIELD, EMPTY_FIELD)
        elif length == 1:
            # One word missing
            (title, title_conf), (name, name_conf) = self.eng_fn_len_1(values)
        else:
            (title, title_conf), (name, name_conf) = self.eng_fn_len_2_more(values, length)

        name = ' '.join(list(filter(lambda w: len(w) > 2 or w == 'Na', name.split())))
        return (title, title_conf), (name, name_conf)

    def thai_get_clean_values(self, img):
        values = self.extract_values(img, 'tha', self.tha_configs)
        values = list(map(NameExtractor.thai_clean_value, values))
        values = list(filter(lambda v: len(v) > 0, values))
        if len(values) == 0:
            return None, 0

        values, length = utils.most_frequent_length(values)
        return values, length

    def thai_len_1(self, values):
        title, sim = self.choose_title(values, 'tha')
        if sim > TITLE_MIN_SIM:
            return ((title, sim), EMPTY_FIELD, EMPTY_FIELD)

        name, ratio = utils.most_frequent(values, with_ratio=True)
        return (EMPTY_FIELD, (name, ratio), EMPTY_FIELD)

    def thai_len_2(self, values, eng_fn_len):
        first_words = utils.map_to_words(values, i=0)
        title, sim = self.choose_title(first_words, 'tha')

        if sim > TITLE_MIN_SIM:
            # Title + firstname
            firstname, ratio = utils.most_frequent(utils.map_to_words(values, i=1), with_ratio=True)
            return ((title, sim), (firstname, ratio), EMPTY_FIELD)
        else:
            # Look for second words, in case of bad croping
            second_words = utils.map_to_words(values, i=1)
            title, sim = self.choose_title(second_words, 'tha')
            if sim > TITLE_MIN_SIM:
                return ((title, sim), EMPTY_FIELD, EMPTY_FIELD)

            # Look if title included in first word
            title, values_no_title = self.find_included_title(values, 'tha')
            values = values if title == '' else values_no_title

            value, ratio = utils.most_frequent_by_word(values)
            if eng_fn_len > 1:
                return ((title, TITLE_MIN_SIM), (value, ratio), EMPTY_FIELD)
            else:
                fn, ln = value.split()
                return ((title, TITLE_MIN_SIM), (fn, ratio), (ln, ratio))

    def thai_len_3_more(self, values, eng_fn_len, eng_ln_len, length):
        first_words = utils.map_to_words(values, i=0)
        title, sim = self.choose_title(first_words, 'tha')

        if sim < TITLE_MIN_SIM:
            # Look for second words, in case of bad croping
            second_words = utils.map_to_words(values, i=1)
            title, sim = self.choose_title(second_words, 'tha')

            if sim > TITLE_MIN_SIM:
                # Remove all the first words which are bullshit
                values = utils.map_to_words(values, start=1)
                length -= 1
            else:
                # Try to form a title from first two words
                merged = list(map(lambda x: x[0] + x[1], zip(utils.map_to_words(values, i=0), utils.map_to_words(values, i=1))))
                title, sim = self.choose_title(merged, 'tha')
                if sim > TITLE_MIN_SIM:
                    values = list(map(lambda x: x[0] + ' ' + x[1], zip(merged, utils.map_to_words(values, start=2))))
                else:
                    # Look if title included in first word
                    title, values_no_title = self.find_included_title(values, 'tha')
                    length += 1
                    if title == '':
                        values = list(map(lambda v: 'TITLE ' + v, values))
                    else:
                        values = list(map(lambda v: title + ' ' + v, values_no_title))

        if (eng_fn_len == 0) or (eng_ln_len == 0) or (eng_fn_len + eng_ln_len != length - 1):
            # Does not sum up correctly with English
            # Take only one first name arbitrarly
            fn_values = utils.map_to_words(values, start=1, end=2)
            ln_values = utils.map_to_words(values, start=2)
        else:
            fn_values = utils.map_to_words(values, start=1, end=1+eng_fn_len)
            ln_values = utils.map_to_words(values, start=1+eng_fn_len)

        firstname, ratio_fn = utils.most_frequent_by_word(fn_values)
        lastname, ratio_ln = utils.most_frequent_by_word(ln_values)

        conf_title = max(sim, TITLE_MIN_SIM)
        return ((title, conf_title), (firstname, ratio_fn), (lastname, ratio_ln))

    def eng_get_clean_correct_values(self, img):
        values = self.extract_values(img, 'eng', self.eng_configs)
        values = list(map(NameExtractor.eng_clean_value, values))
        values = list(filter(lambda v: len(v) > 0, values))
        if len(values) == 0:
            return None, 0

        values, length = utils.most_frequent_length(values)
        values = NameExtractor.eng_correct_words(values)
        return values, length

    def eng_fn_len_1(self, values):
        title, sim = self.choose_title(values, 'eng')
        if sim > TITLE_MIN_SIM:
            return ((title, sim), EMPTY_FIELD)
        else:
            name, ratio = utils.most_frequent(values, with_ratio=True)
            name = ' '.join(list(filter(lambda w: len(w) > 2 or w == 'Na', name.split())))
            return (EMPTY_FIELD, (name, ratio))

    def eng_fn_len_2_more(self, values, length):
        first_words = utils.map_to_words(values, i=0)
        title, sim = self.choose_title(first_words, 'eng')

        values = utils.map_to_words(values, start=1)
        if length == 2:
            name, ratio = utils.most_frequent(values, with_ratio=True)
        else:
            name, ratio = utils.most_frequent_by_word(values)

        name = ' '.join(list(filter(lambda w: len(w) > 2 or w == 'Na', name.split())))
        return ((title, sim), (name, ratio))

    def match_titles(self, res):
        try:
            eng_to_tha = self.title_df[self.title_df.eng == res['eng_title']].tha.values[0]
            tha_to_eng = self.title_df[self.title_df.tha == res['tha_title']].eng.values[0]
        except:
            print('One of the proposed titles does not exist, aborting matching')
            return res

        if res['tha_title'] == eng_to_tha:
            return res

        eng_conf, tha_conf = res['eng_title_conf'], res['tha_title_conf']
        if eng_conf > tha_conf:
            res['tha_title'] = eng_to_tha
        else:
            tha_to_eng = self.title_df[self.title_df.tha == res['tha_title']].eng.values[0]
            res['eng_title'] = tha_to_eng

        return res

    def choose_title(self, titles, lang):
        labels_titles = self.title_df[lang].values
        most_similar, best_score = '', 0

        for title in titles:
            if title in labels_titles:
                return title, 1

            for label_title in labels_titles:
                sim = NameExtractor.title_sim(title, label_title, lang)

                if sim > best_score:
                    best_score = sim
                    most_similar = label_title

        return most_similar, best_score

    @staticmethod
    def title_sim(candidate, label, lang):
        candidate = candidate.replace('.', '')
        label = label.replace('.', '')
        if len(label) > 3 or len(candidate) != len(label) or lang == 'eng':
            return utils.similarity(candidate, label)

        count = utils.count_same_chars(candidate, label)
        if count == len(label):
            return 1
        elif count == len(label) - 1:
            return 0.9
        return 0

    def find_included_title(self, values, lang):
        threshold = int(INCLUDED_TITLE_RATIO*len(values))
        for label_title in self.title_df[lang].values:
            values_with_title = []
            for value in values:
                label_title_nodot = label_title.replace('.', '')
                value_nodot = value.replace('.', '')
                end = min(len(label_title_nodot), len(value_nodot))
                sim = NameExtractor.title_sim(value_nodot[:end], label_title_nodot, lang)
                if sim > TITLE_MIN_SIM:
                    if value[end] == '.':
                        end += 1
                    values_with_title.append(value_nodot[end:])

            if len(values_with_title) > threshold:
                return label_title, values_with_title

        return '', None

    def extract_values(self, img, lang, tesseract_confs):
        look = lambda scale, config: pytesseract.image_to_string(utils.resize(img, scale), lang, config=config)
        return [look(scale, conf) for scale in self.scaling_range for conf in tesseract_confs]

    def google_tha_to_eng(self, word):
        try:
            return self.translator.translate(word, src='th', dest='en').text
        except:
            return ''

    def google_eng_to_tha(self, word):
        try:
            chars = [self.translator.translate(c, src='en', dest='th').text for c in word]
            return utils.filter_english_letters(''.join(chars))
        except:
            return ''

    def google_tha_to_tha(self, word):
        try:
            return self.translator.translate(word, src='th').extra_data['possible-mistakes'][1]
        except:
            return word

    def google_eng_to_eng(self, word):
        try:
            return self.translator.translate(word, src='en').extra_data['possible-mistakes'][1]
        except:
            return word

    def fill_by_default(self, res):
        if res['tha_title'] == '' and res['eng_title'] == '':
            thai = NameExtractor.get_static_translation(self.title_df, DEFAULT_TITLE, 'eng', 'tha')
            res['eng_title'] = DEFAULT_TITLE
            res['tha_title'] = thai

        return res

    def balance_thai_name(self, tha_fn, tha_ln):
        if len(tha_fn) <= 2 and len(tha_ln.split()) > 1:
            tha_fn = tha_fn + ' ' + tha_ln.split()[0]
            tha_ln = ' '.join(tha_ln.split()[1:])
        return tha_fn, tha_ln

    def translate_missing(self, res):
        trans_static = NameExtractor.get_static_translation
        if res['tha_title'] == '' and res['eng_title'] != '':
            res['tha_title'] = trans_static(self.title_df, res['eng_title'], 'eng', 'tha')
            if GOOGLE_ENG_TO_THA and res['tha_title'] == '':
                res['tha_title'] = self.google_eng_to_tha(res['eng_title'])

        elif res['eng_title'] == '' and res['tha_title'] != '':
            res['eng_title'] = trans_static(self.title_df, res['tha_title'], 'tha', 'eng')
            if res['eng_title'] == '':
                res['eng_title']  = NameExtractor.romanize(res['tha_title'])


        if res['tha_firstname'] == '' and res['eng_firstname'] != '':
            res['tha_firstname'] = trans_static(self.fistname_df, res['eng_firstname'], 'eng', 'tha')
            if GOOGLE_ENG_TO_THA and res['tha_firstname'] == '':
                res['tha_firstname'] = self.google_eng_to_tha(res['eng_firstname'])

        elif res['eng_firstname'] == '' and res['tha_firstname'] != '':
            res['eng_firstname'] = trans_static(self.fistname_df, res['tha_firstname'], 'tha', 'eng')
            if res['eng_firstname'] == '':
                res['eng_firstname']  = NameExtractor.romanize(res['tha_firstname'])


        if res['tha_lastname'] == '' and res['eng_lastname'] != '':
            res['tha_lastname'] = trans_static(self.lastname_df, res['eng_lastname'], 'eng', 'tha')
            if GOOGLE_ENG_TO_THA and res['tha_lastname'] == '':
                res['tha_lastname'] = self.google_eng_to_tha(res['eng_lastname'])

        elif res['eng_lastname'] == '' and res['tha_lastname'] != '':
            res['eng_lastname'] = trans_static(self.lastname_df, res['tha_lastname'], 'tha', 'eng')
            if res['eng_lastname'] == '':
                res['eng_lastname']  = NameExtractor.romanize(res['tha_lastname'])

        return res

    @staticmethod
    def romanize(value):
        name = ''
        try:
            name = romanization(value)
        except:
            try:
                name =  self.google_tha_to_eng(value)
            except:
                pass

        return utils.keep_english_letters(name.title())

    @staticmethod
    def thai_clean_value(value):
        blacklist = '!-/\_"\'“0123456789[];()\{\}€”<>+=*:|`@?#°»฿%×๐'
        value = value.replace('\n', ' ').replace(',', '.').replace("ร.น", "ร.น.").replace("..", ".")
        chars = list(filter(lambda c: c not in blacklist, value))
        chars = ''.join(chars).strip()
        words = chars.split()
        words = list(filter(lambda w: w not in '.,', words))
        words = list(map(lambda w: w[1:] if w.startswith('.') else w, words))
        words = [NameExtractor.rmdiacritics(w[0]) + w[1:] if type(w) == str and len(w) > 0 else w for w in words]
        if len(words) > 1:
            # Cropping noise
            words = words[1:] if len(words[0]) == 1 and len(words[1]) > 1 else words
            words = words[:-1] if len(words[-1]) == 1 and len(words[1]) > 1 else words

            # Combine 2 words title
            if NameExtractor.title_sim(words[0], 'ว่าที่', 'tha') > 0.65 and len(words[1]) < 5:
                words = [words[0] + words[1]] + words[2:]

        return ' '.join(words)

    @staticmethod
    def rmdiacritics(char):
        '''
        Return the base character of char, by "removing" any
        diacritics like accents or curls and strokes and the like.
        '''
        search_string = ''.join((c for c in unicodedata.normalize('NFD', char) if unicodedata.category(c) != 'Mn'))
        return search_string

    @staticmethod
    def eng_clean_value(value):
        value = unidecode.unidecode(value)
        value = value.replace('\n', ' ')
        value = value.replace('eee', 'ee')
        chars = list(filter(lambda c: c.isalpha() or c == ' ', value))
        value = ''.join(chars)
        value = utils.correct_caps(value)
        words = value.split()
        words = list(filter(lambda w: len(w) > 1, words))
        words = list(map(lambda w: w.strip(), words))
        words = list(map(NameExtractor.hard_correct_word, words))
        words = NameExtractor.combine_english_title(words)
        return ' '.join(words)

    @staticmethod
    def get_static_translation(df, ref, ref_lang, target_lang):
        candidates = df[df[ref_lang] == ref][target_lang].values
        return candidates[0] if len(candidates) > 0 else ''

    @staticmethod
    def validate_tha_models(models):
        for i, model in enumerate(models):
            if THA_MODEL_REQUIRED_FLAG not in model:
                models[i] += (' ' + THA_MODEL_REQUIRED_FLAG)
        return models

    @staticmethod
    def eng_correct_words(values):
        correct = lambda v: NameExtractor.correct_value(v, values)
        return list(map(correct, values))

    @staticmethod
    def correct_value(value, values):
        yes, letters = utils.has_three_consecutive_consonnants(value, exception=CONSONNANTS_EXCPETIONS)
        if not yes:
            return value

        for c in letters:
            c_low = c.islower()
            c = c.lower()
            if c in LETTER_CORRECTION:
                replacement = LETTER_CORRECTION[c] if c_low else LETTER_CORRECTION[c].upper()
                corrected_letters = letters.replace(c, replacement)
                corrected_value = value.replace(letters, corrected_letters)
                if corrected_value in values:
                    return corrected_value

        return value

    @staticmethod
    def hard_correct_word(w):
        return HARD_CORRECTION[w] if w in HARD_CORRECTION else w

    @staticmethod
    def combine_english_title(words):
        if len(words) < 2:
            return words

        w1 = words[0].lower()
        w2 = words[1].lower()
        if len(words) < 3:
            if len(w1) < 4 and utils.count_letters_in('lt', w1) > 0:
                if len(w2) < 5 and (utils.count_letters_in('jg', w2) > 0 or utils.count_letters_in('col', w2) > 1):
                    return [words[0] + words[1]] + words[2:]
        else:
            w3 = words[2].lower()
            if utils.similarity('acting', w1) > TITLE_MIN_SIM:
                if len(w2) < 4 and utils.count_letters_in('sub', w2) > 2:
                    if len(w3) < 3 and utils.count_letters_in('lt', w3) > 0:
                        return [words[0] + words[1] + words[2]] + words[3:]

        return words