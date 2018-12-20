import random, re
import pandas as pd
from PIL import Image
from collections import OrderedDict

from address_extraction import extract_addresses
from address_accuracy_computation import compute_metrics

LABELS_PATH = 'data/DIE_Train_Label_3Scenario.xlsx'
SYMBOLS_PATH = 'address_extraction/resources/symbols.csv'
COMPONENTS_PATH = 'address_extraction/resources/components.csv'
IMG_PATH = 'data/Address/{:0=3d}/6_field_img_denoised_address_l{}.TIFF'

def validate(img_id_start, img_id_end=None):
    img_id_end = img_id_end if img_id_end else img_id_start + 1
    
    labels = pd.read_excel(LABELS_PATH, sheet_name='Address')
    symbols = pd.read_csv(SYMBOLS_PATH)
    components = pd.read_csv(COMPONENTS_PATH)

    r = list(range(img_id_start, img_id_end))
    #random.shuffle(r)
    true_list = []
    res_list = []
    ids_list = []
    comment_list = []
    for img_id in r:

        try:
            l1_img = Image.open(IMG_PATH.format(img_id, 1))
            l2_img = Image.open(IMG_PATH.format(img_id, 2))
        except FileNotFoundError:
            #print('ID {} not found'.format(img_id))
            continue

        print('Processing ID {}'.format(img_id))
        dict_ = OrderedDict([
            ('address_l1', l1_img),
            ('address_l2', l2_img)
        ])
        res = extract_addresses(dict_)
        true = label(img_id, labels)

        res_list.append(res['address_l1'] + ' ' + res['address_l2'])
        true_list.append(true['address_l1'] + ' ' + true['address_l2'])
        ids_list.append(img_id)
        comment_list.append("")

        print("*"*5)
        print("res:", res['address_l1'] + ' ' + res['address_l2'])
        print("true:", true['address_l1'] + ' ' + true['address_l2'])
        print()

    compute_metrics(true_list, res_list, ids_list=ids_list)

def score(img_id, res, true, symbols, components):
    symbols = symbols.set_index('name').squeeze().to_dict()
    components = components.set_index('name').squeeze().to_dict()

    exact_match(res, true)

    if components['bangkok'] in true['address_l2'] or res['bangkok'][1] != '':
        eval_bangkok(res, true, symbols, components)
    else:
        eval_non_bangkok(res, true, symbols, components)

    print()

def eval_bangkok(res, true, symbols, components):    
    eval_component(res, true, 'bangkok', components['bangkok'], check_next=False)
    eval_component(res, true, 'khet', components['khet'], check_next=False)
    eval_component(res, true, 'khwaeng', components['khwaeng'], check_next=True)
    eval_symbol(res, true, 'something', symbols['something'])

def eval_non_bangkok(res, true, symbols, components):
    pass

def eval_component(res, true, name, comp, check_next=True):
    true_addr = ' '.join([true['address_l1'], true['address_l2']])
    res_addr = ' '.join([res['address_l1'], res['address_l2']])

    present, true_val, true_prev, true_next = search_component(true_addr, comp)
    if not present:
        # Do not evaluate if not present
        return

    present, res_val, res_prev, res_next = search_component(res_addr, comp)
    print('\tCompononent {}({}):'.format(name, comp))
    print('\t\tComponent found: {}'.format(present))
    if present:
        print('\t\tValue checking:  {}: {:<10} vs {:<10}'.format(res_val == true_val, res_val, true_val))
        if name == 'city_num':
            print('\t\tNumber match prev number: {} {:<6} vs {:<6}, next number: {} {:<6} vs {:<6}'.format(
                res_prev == true_prev, res_prev, true_prev, res_next == true_next, res_next, true_next))

def eval_symbol(res, true, name, symbol):
    true_addr = ' '.join([true['address_l1'], true['address_l2']])
    res_addr = ' '.join([res['address_l1'], res['address_l2']])

    present, true_value, true_prev, true_next = search_symbol(true_addr, symbol)
    if not present:
        # Do not eval if not present in label
        return
    
    present, res_value, res_prev, res_next = search_symbol(res_addr, symbol)

    print('\tSymbol {}({}):'.format(name, symbol))
    print('\t\tSymbol found: {}'.format(present))
    if present:
        print('\t\tValue match:  {}: {:<10} vs {:<10}'.format(res_value == true_value, res_value, true_value))

def exact_match(res, true):
    res_l1, res_l2 = res['address_l1'], res['address_l2']
    true_l1, true_l2 = true['address_l1'], true['address_l2']
    print('\tL1 exact match: {}: {:<50} vs {:<50}'.format(res_l1 == true_l1, res_l1, true_l1))
    print('\tL2 exact match: {}: {:<50} vs {:<50}'.format(res_l2 == true_l2, res_l2, true_l2))

def label(img_id, labels):
    values = labels[labels['Filename'] == img_id].to_dict()
    return OrderedDict([
        ('address_l1', list(values['Address line 1'].values())[0]),
        ('address_l2', list(values['Address line 2'].values())[0])
    ])

if __name__=='__main__':
    validate(567)
