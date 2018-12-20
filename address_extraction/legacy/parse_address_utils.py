try:
    import address_extraction.address_utils as address_utils
except ModuleNotFoundError:
    import address_utils

def find_district_province(words, districts, provinces, nb_results):
    # found_district_province = []

    # if len(words) >= 3:
    #     f_provinces = find_province(words, provinces, nb_results)
    #     f_districts = find_district(words, districts, nb_results)

    #     if f_provinces is not None and f_districts is not None:
    #         for f_p in f_provinces:
    #             for f_d in f_districts:
    #                 case = {}
    #                 case['similarity'] = compound_similarity(f_d[0][1], f_p[0][1])
    #                 case['district'] = f_d[0][0]
    #                 case['province'] = f_p[0][0]
    #                 case['parsed'] = f_d[1]
    #                 found_district_province.append(case)

    #         return found_district_province

    #     elif f_provinces is not None:
    #         for f_p in f_provinces:
    #             case = {}
    #             case['similarity'] = f_p[0][1]
    #             case['district'] = ""
    #             case['province'] = f_p[0][0]
    #             case['parsed'] = f_p[1]
    #             found_district_province.append(case)

    #     elif f_districts is not None:
    #         for f_d in f_districts:
    #             case = {}
    #             case['similarity'] = f_d[0][1]
    #             case['province'] = ""
    #             case['district'] = f_d[0][0]
    #             case['parsed'] = f_d[1]
    #             found_district_province.append(case)

    #     else:
    #         return None

    # else:
    #     return None

    f_provinces = find_province(words, provinces, nb_results)

    print('Extracted province: ', f_provinces)
    # f_districts = find_district(words, districts, nb_results)


def find_province(words, provinces, nb_results):
    
    if words is None:
        return None
    else:

        best_match = {
            'score': 0,
            'word': ''
        }
        for word in words[-3]:
            _, match_score = word_matching(provinces, word)
            if match_score > best_match['score']:
                best_match['score']=match_score
                best_match['word']=word

        return best_match

    # # 1. check if last word is province
    # matches_1 = [(w_m, 1) for w_m in word_matching(
    #     provinces, words[-1])]

    # # 2. check if last 2 words is province
    # if len(words) >= 2:
    #     matches_2 = [(w_m, 2) for w_m in word_matching(
    #         provinces, "".join(words[-2:]))]

    # # sort by highest similarity
    # options = matches_1 + matches_2

    # if len(options):
    #     max_result = sorted(options, key=lambda tup: -tup[0][1])
    #     return max_result[:nb_results]
    # else:
    #     return None

def find_district(words, districts, nb_results):

    if words is None:
        return None
    else:

        best_match = {
            'score': 0,
            'word': ''
        }
        for word in words[-3]:
            _, match_score = word_matching(districts, word)
            if match_score > best_match['score']:
                best_match['score']=match_score
                best_match['word']=word

        return best_match

    # if not len(words):
    #     return None

    # # 1. check if second last word is district
    # if len(words) >= 2:
    #     matches_1 = [(w_m, 2) for w_m in word_matching(
    #         districts, words[-2])]

    # # 2. check if third last word is district
    # if len(words) >= 3:
    #     matches_2 = [(w_m, 3) for w_m in word_matching(
    #         districts, words[-3])]

    # # 3. check if second last and third last words is district
    # if len(words) >= 3:
    #     matches_3 = [(w_m, 3) for w_m in word_matching(
    #         districts, "".join(words[-3:-1]))]

    # # sort by highest similarity
    # options = matches_1 + matches_2 + matches_3

    # if len(options):
    #     max_result = sorted(options, key=lambda tup: -tup[0][1])
    #     return max_result[:nb_results]
    # else:
    #     return None


def select_most_likely_district_province(candidates, valid_districts_provinces):
    '''candidates are of the form:
    {similarity:float, district:str, province:str, parsed:int}
    '''

    def is_valid_c(c, df=valid_districts_provinces):
        i_df = df[(df.district == c['district']) & (df.province == c['province'])]
        return len(i_df) > 0

    valid_candidates = [c for c in candidates if is_valid_c(c)]
    # what if valid_candidates is empty?

    if len(valid_candidates):
        max_result = sorted(valid_candidates, key=lambda dic: -dic['similarity'])
        return max_result[0]

    else:
        return None