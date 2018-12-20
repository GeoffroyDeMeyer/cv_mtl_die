from difflib import SequenceMatcher


def clean_text(text):
    '''Remove punctuations from extracted name'''
    text = text.replace('\n', ' ')
    # remove punctuation
    punctuations = "!?@#$%^&*()_+=,.<>\|™;¥~:©®¢\[\]\{\}«“\"\‘\”\'\’"
    cleaned_text = ''.join(ch for ch in text if ch not in punctuations)

    return cleaned_text


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def compound_similarity(a, b):
    if (a <= 0) or (b <= 0):
        return 0
    else:
        return 1/(1/a + 1/b)

def word_matching(candidates, word, ratio_threshold=0):
    '''Find the most likely candidate similar to word from a list of candidates.'''

    # calculate similarity between each candidate and the word of interest
    candidates_and_sim_score = [(c, similar(word, c)) for c in candidates]

    # sort by the highest similarity
    candidates_and_sim_score_sorted = sorted(candidates_and_sim_score, key=lambda tup: -tup[1])
    # print(candidates_and_sim_score_sorted)

    # check if ratio is higher than min
    if candidates_and_sim_score_sorted[0][1] >= ratio_threshold:
        best_candidate = candidates_and_sim_score_sorted[0][0]
        best_similarity_score = candidates_and_sim_score_sorted[0][1]
        return (best_candidate, best_similarity_score)
    else:
        return ()


def separate_lines(text, l1, l2):
    print()
    print("address_utils.separate_lines is not implemented")
    print()
    return text, ""
