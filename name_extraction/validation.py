import random
import pandas as pd
from PIL import Image

from collections import OrderedDict
from difflib import SequenceMatcher
from name_extraction import extract_names

LABELS_PATH = 'data/DIE_Train_Label_3Scenario.xlsx'
IMG_PATH = 'data/Name/{:0=3d}/6_field_img_denoised_{}.TIFF'

sim = {}

def validate(img_id_start, img_id_end=None):
    img_id_end = img_id_end if img_id_end else img_id_start + 1
    
    labels = pd.read_excel(LABELS_PATH).fillna('')

    r = list(range(img_id_start, img_id_end))
    random.shuffle(r)
    for img_id in r:
        try:
            thai_img = Image.open(IMG_PATH.format(img_id, 'name_tha'))
            eng_fn_img = Image.open(IMG_PATH.format(img_id, 'name_eng'))
            eng_ln_img = Image.open(IMG_PATH.format(img_id, 'last_name_eng'))
        except FileNotFoundError:
            #print('ID {} not found'.format(img_id))
            continue

        print('Processing ID {}'.format(img_id))
        imgs = OrderedDict([
            ('name_tha', thai_img),
            ('name_eng', eng_fn_img),
            ('last_name_eng', eng_ln_img)
        ])
        res = extract_names(imgs)
        true = label(img_id, labels)
        score(img_id, res, true)

    df = pd.DataFrame.from_dict(sim, orient='index')
    final_score(df)
        
def label(img_id, labels):
    values = labels[labels['Filename'] == img_id].to_dict()
    return OrderedDict([
        ('tha_title',        list(values['Title (TH)'].values())[0]),
        ('tha_firstname',    list(values['First name (TH)'].values())[0]),
        ('tha_lastname',     list(values['Last name (TH)'].values())[0]),
        ('eng_title',        list(values['Title (EN)'].values())[0]),
        ('eng_firstname',    list(values['First name (EN)'].values())[0]),
        ('eng_lastname',     list(values['Last name (EN)'].values())[0])
    ])

def sim_score(a, b):
    return SequenceMatcher(a=a, b=b).ratio()

def score(img_id, result, label):
    sim[img_id] = {}
    for key, value in result.items():
        if 'conf' in key:
            continue

        sim[img_id][key] = sim_score(value, label[key])
        print('\t{:<15}: {:<5.2f} {:<30} {:<30}'.format(key, sim[img_id][key], label[key], value))
        
    print()

def final_score(df):
    print('Mean similarity:')
    print(df.mean(axis=0))
    print()
    print('Accuracy:')
    print(df.apply(lambda x: x == 1).sum() / len(df))

if __name__=='__main__':
    validate(117, 118)
