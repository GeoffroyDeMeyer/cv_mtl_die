'''
Utils function used for name extraction.
'''

import re
import numpy as np
from collections import Counter
from difflib import SequenceMatcher

def most_frequent_by_word(values):
    '''Extract most frequent word for each position in the list of string'''
    winners = []
    ratios = []
    lengths = map(lambda v: len(v.split()), values)
    for i in range(min(lengths)):
        candidates = map_to_words(values, i=i)
        winner, ratio = most_frequent(candidates, with_ratio=True)
        winners.append(winner)
        ratios.append(ratio)

    return ' '.join(winners), np.mean(ratios)

def most_frequent_length(values):
    '''Compute most frequent length word wise in all the list of string'''
    lengths = [len(value.split()) for value in values]
    length = most_frequent(lengths)
    filtered = list(filter(lambda v: len(v.split()) == length, values))
    return filtered, length

def map_to_words(values, i=None, start=None, end=None):
    '''Select specific words in each string'''
    if i is None:
        if end is None:
            return [' '.join(value.split()[start:]) for value in values]
        else:
            return [' '.join(value.split()[start:end]) for value in values]
    else:
        return [value.split()[i] for value in values]

def resize(img, scale):
    '''Rescale image'''
    w, h = img.size
    return img.resize((int(w*scale), int(h*scale)))

def most_frequent(lst, with_ratio=False):
    '''Return most frequent value'''
    most_frequent = Counter(lst).most_common(1)[0]
    if with_ratio:
        ratio = most_frequent[1] / len(lst)
        return most_frequent[0], ratio
    else:
        return most_frequent[0]

def similarity(a, b):
    '''Compute string similarity'''
    return SequenceMatcher(a=a, b=b).ratio()

def correct_caps(v):
    '''Correct capitalisation'''
    # Split by caps
    words = re.findall('[A-Z][^A-Z]*', v)
    words = list(filter(is_capitalized, words))
    return ' '.join(words)

def is_capitalized(w):
    '''True if word is capitalised'''
    if len(w) == 1:
        return w.isupper()
    else:
        return w[0].isupper() and w[1:].islower()

def count_same_chars(s1, s2):
    count = 0
    for i, c in enumerate(s1):
        if c == s2[i]:
            count+=1

    return count
    

def has_three_consecutive_consonnants(s, exception=[]):
    windows = [s[i:i+3] for i in range(len(s) - 2)]
    for w in windows:
        if w.lower() not in exception and all(map(is_consonnant, w)):
            return True, w

    return False, ''

def is_consonnant(c):
    return c.isalpha() and c.lower() not in 'aeiou'

def count_letters_in(a, b):
    count = 0
    for c in a:
        if c in b:
            count += 1

    return count

def keep_english_letters(w):
    f = lambda c: is_english_letter(c) or c == ' '
    return ''.join(list(filter(f, w)))

def filter_english_letters(w):
    f = lambda c: not is_english_letter(c) or c == ' '
    return ''.join(list(filter(f, w)))

def is_english_letter(c):
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWYZabcdefghijklmnopqrstuvwxyz'
    return c in alphabet