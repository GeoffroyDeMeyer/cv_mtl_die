import pandas as pd
import numpy as np
import sys

from difflib import SequenceMatcher
from collections import OrderedDict

try:
    from address_extraction.address_extractor import AddressExtractor
except ImportError:
    from address_extractor import AddressExtractor



RESOURCE_PATH   = 'address_extraction/resources/'

symbol_dict = OrderedDict(sorted(pd.read_csv(RESOURCE_PATH + 'symbols.csv').set_index('name').squeeze().to_dict().items(), key=lambda t: t[0]))
component_dict = OrderedDict(sorted(pd.read_csv(RESOURCE_PATH + 'components.csv').set_index('name').squeeze().to_dict().items(), key=lambda t: t[0]))

ae = AddressExtractor()

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def extract_xlsx_sheet(path, sheet):
    return pd.read_excel(path, sheet_name=sheet).fillna("")

def extract_xlsx(path):
    return (extract_xlsx_sheet(path, 0), extract_xlsx_sheet(path, 1), extract_xlsx_sheet(path, 2))

def has_digit(inputString):
    return any(char.isdigit() for char in inputString)

def compute_address_similarity(true_addresses, res_addresses, ids_list=None):

    per_symbol_ratio_dict = OrderedDict([(symbol, {'field': [], 'field_value': []}) for symbol in symbol_dict.keys()])
    per_component_ratio_dict = OrderedDict([(component, {'field': [], 'field_value': []}) for component in component_dict.keys()])
    per_component_ratio_dict["number1"] = {'field': [], 'field_value': []}
    per_component_ratio_dict["number2"] = {'field': [], 'field_value': []}


    true_labels = []
    true_idxs = []
    res_idxs = []
    comment_list = []

    for true_address, res_address in list(zip(true_addresses, res_addresses)):
        index_true = ae.extract_index(true_address)
        index_res = ae.extract_index(res_address)

        # per symbol ratio
        for symbol, _ in per_symbol_ratio_dict.items():
            if symbol in index_true:
                if symbol in index_res:
                    per_symbol_ratio_dict[symbol]["field"].append(1)
                    per_symbol_ratio_dict[symbol]["field_value"].append(similar(index_true[symbol][1], index_res[symbol][1]))
                else:
                    per_symbol_ratio_dict[symbol]["field"].append(0)
                    per_symbol_ratio_dict[symbol]["field_value"].append(0)


        # per component ratio
        for component, _ in per_component_ratio_dict.items():
            if component in index_true:
                if component in index_res:
                    per_component_ratio_dict[component]["field"].append(1)
                    per_component_ratio_dict[component]["field_value"].append(similar(index_true[component][1], index_res[component][1]))
                else:
                    per_component_ratio_dict[component]["field"].append(0)
                    per_component_ratio_dict[component]["field_value"].append(0)
 
        # prepare column to save in csv
        comment_list.append("")
        true_labels.append(true_address)
        for k, v in index_true.copy().items():
            if k in index_res and v[1] == index_res[k][1]:
                del(index_true[k])
                del(index_res[k])
        true_idxs.append(index_true)
        res_idxs.append(index_res)

    # save to csv      
    save = pd.DataFrame()
    if ids_list != None:
        save["filename"] = ids_list
        save = save.set_index("filename")
    save["true label"] = true_labels 
    save["true index"] = true_idxs
    save["res index"] = res_idxs
    save["comment"] = comment_list
    save.to_csv("results.csv") 

    return per_symbol_ratio_dict, per_component_ratio_dict


def extract_address_sheet(original_path, extracted_path):

    _, _, true_address_sheet = extract_xlsx(original_path)
    _, _, res_address_sheet = extract_xlsx(extracted_path)

    res_address_sheet[res_address_sheet.columns[1]] = res_address_sheet[res_address_sheet.columns[1]].astype(str)
    res_address_sheet[res_address_sheet.columns[2]] = res_address_sheet[res_address_sheet.columns[2]].astype(str)
    
    true_address = true_address_sheet[true_address_sheet.columns[1:3]].apply(lambda x: ' '.join(x), axis=1)
    res_address = res_address_sheet[res_address_sheet.columns[1:3]].apply(lambda x: ' '.join(x), axis=1)

    return true_address, res_address

def compute_metrics(true_address, res_address, ids_list=None):

    per_symbol_ratio_dict, per_component_ratio_dict = compute_address_similarity(true_address, res_address, ids_list)

    exact_match_ratios = []
    similarity_match_ratios = []
    metrics_per_field = {}
    print()
    print(10*'*')
    print('Per Symbol Ratio')
    print()
    for k,v in per_symbol_ratio_dict.items():
        if len(v['field']) != 0:
            exact_match_field_ratio = np.mean([1 if x==1 else 0 for x in v['field_value']])*100
            # weighted mean
            for i in range(len(v['field'])):
                exact_match_ratios.append(exact_match_field_ratio)

            similarity_match_field_value_ratio = np.mean(v['field_value'])*100
            for i in range(len(v['field'])):
                similarity_match_ratios.append(similarity_match_field_value_ratio)

            print(' - {:<1} ({:<9}) \t total symbol : {:<3}\t detection of symbol : {:05.2f}% (exact match) \t detection of symbol\'s value: {:05.2f}% (similarity) {:05.2f}% (exact match)'
            .format(symbol_dict[k], k, len(v['field']), np.mean(v['field'])*100, similarity_match_field_value_ratio, exact_match_field_ratio))

            metrics_per_field["Address " + k] = (exact_match_field_ratio, similarity_match_field_value_ratio)

    print()
    print(10*'*')
    print('Per Component Ratio')
    print()
    for k,v in per_component_ratio_dict.items():
        if len(v['field']) != 0:
            exact_match_field_ratio = np.mean([1 if x==1 else 0 for x in v['field_value']])*100
            # weighted mean
            for i in range(len(v['field'])):
                exact_match_ratios.append(exact_match_field_ratio)

            similarity_match_field_value_ratio = np.mean(v['field_value'])*100
            for i in range(len(v['field'])):
                similarity_match_ratios.append(similarity_match_field_value_ratio)

            print(' - {:<17} \t total symbol : {:<3}\t detection of component: {:05.2f}% (exact match) \t detection of component\'s value: {:05.2f}% (similarity) {:05.2f}% (exact match)'
            .format(k, len(v['field']), np.mean(v['field'])*100, similarity_match_field_value_ratio, exact_match_field_ratio))

            metrics_per_field["Address " + k] = (exact_match_field_ratio, similarity_match_field_value_ratio)

    print()
    print(10*'*')
    print('Exact match ratio (fields value): {:05.2f}%'.format(np.mean(exact_match_ratios)))
    print('Similarity match ratio (fields value): {:05.2f}%'.format(np.mean(similarity_match_ratios)))

    return np.mean(exact_match_ratios), np.mean(similarity_match_ratios), metrics_per_field

if __name__ == '__main__':

    ORIGINAL_DATA_PATH = sys.argv[1]
    EXTRACTED_DATA_PATH = sys.argv[2]

    true_address, res_address = extract_address_sheet(ORIGINAL_DATA_PATH, EXTRACTED_DATA_PATH)
    compute_metrics(true_address, res_address)
