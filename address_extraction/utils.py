import re
from itertools import chain
from collections import Counter, defaultdict
from difflib import SequenceMatcher

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def most_frequent_length(values):
    '''Compute most frequent length word wise in all the list of string'''
    lengths = [len(value.split()) for value in values]
    length = most_frequent(lengths)
    filtered = list(filter(lambda v: len(v.split()) == length, values))
    return filtered, length

def most_frequent_amount_figures(values):
    counts = [count_figures(value) for value in values]
    count = most_frequent(counts)
    filtered = list(filter(lambda v: count_figures(v) == count, values))
    return filtered, count

def most_frequent_words(list_text):
    res = []
    num_words = len(list_text[0].split())
    for i in range(num_words):
        i_words = []
        for text in list_text:
            i_words.append(text.split()[i])
        res.append(most_frequent(i_words))
    
    return ' '.join(res)

def most_frequent_by_key(dicts):
    index = defaultdict(list)
    c_i = defaultdict(list)
    for field, v in chain(*[d.items() for d in dicts]):
        index[field].append(v)
    for field, candidates in index.items():
        candidate, confidence_interval_score = most_frequent_with_c_i(candidates)
        index[field] = candidate
        c_i[field] = confidence_interval_score*100

    return dict(index), dict(c_i)

def count_figures(s):
    r = '\d+/\d+|\d+'
    return len(re.findall(r, s))

def is_figure(s):
    return re.match('\d+/\d+|\d+', s) != None

def most_frequent_with_c_i(lst):
    '''Return most frequent value'''

    best = Counter(lst).most_common()

    if len(best) == 0:
        return '', 0
    else:
        candidate = best[0][0]
        c_i = best[0][1]/len(lst)
        return candidate, c_i

def most_frequent(lst):
    '''Return most frequent value'''
    max_count = 0
    max_length = 0
    for length, count in Counter(lst).most_common():
        if count > max_count:
            max_count = count
            max_length = length
        elif count == max_count and length > max_length:
            max_length = length
    return max_length

def resize(img, scale):
    '''Rescale image'''
    w, h = img.size
    return img.resize((int(w*scale), int(h*scale)))

def find_index(l, lookup_symbol):
    """Return the index of the lookup_symbol from the list l (-1 if it is not found)."""
    index = -1
    for i, e in enumerate(l):
        if e == lookup_symbol:
            index = i
    return index

def idx_starting_with(lst, lookup, min_size=3):
    take_next = False
    for i, e in enumerate(lst):
        if take_next:
            # we take the next value only if big enough
            if len(e) >= min_size: 
                return i, e
            else:
                take_next = False
        if e.startswith(lookup):
            if e == lookup:
                # we found the the symbol, we take the next value
                take_next = True
            else:
                return i, e
    return -1, ''

def idx_similar_with(lst, lookup, threshold):
    for i, e in enumerate(lst):
        if similar(e, lookup) > threshold:
            return i, e
    return -1, ''

def idx_starting_similar_with(lst, lookup, threshold=None, letter_nb=None):
    for i, e in enumerate(lst):
        end = min(len(e), len(lookup))
        if threshold:
            if similar(e[:end], lookup) > threshold:
                return i, e
        else:
            count = 0
            for a, b in list(zip(e, lookup)):
                count = count + 1 if a == b else count
            if count >= letter_nb:
                return i, e
    return -1, ''

def word_matching(candidates, word, ratio_threshold=0):
    """Find the most likely candidate similar to word from a list of candidates."""

    best = ()
    for c in candidates:
        sc = similar(word, c)
        if sc >= ratio_threshold:
            if len(best) == 0 or best[1] < sc:
                best = (c, sc)

    return best

def clean_number(number_string):
    return re.sub(r'[^0-9-/]', '', number_string)

def search_component(s, comp):
    words = s.split()

    for i, w in enumerate(words):
        if w == comp or w.startswith(comp):
            previous = ''
            previous = words[i-1] if i > 0 else previous
            next_ = ''
            next_ = words[i+1] if i < len(words)-1 else next_
            return True, w, previous, next_

    return False, '', '', ''

def search_symbol(s, symbol):
    words = s.split()

    for i, w in enumerate(words):
        if w.startswith(symbol + '.'):
            value = w.split('.')[1]
            previous = ''
            previous = words[i-1] if i > 0 else previous
            next_ = ''
            next_ = words[i+1] if i < len(words)-1 else next_
            return True, value, previous, next_

    return False, '', '', ''

def invert_dict(d):
   inverse = dict()
   for key in d:
       # Go through the list that is saved in the dict:
       for item in d[key]:
           # Check if in the inverted dict the key exists
           if item not in inverse:
               # If not create a new list
               inverse[item] = [key]
           else:
               inverse[item].append(key)
   return inverse