import requests
import wikipedia as w
from bs4 import BeautifulSoup as bs
import re

import pandas as pd
from tqdm import tqdm

import sys

def remove_tags(info, remove_p=False):
    if remove_p:
        return re.sub('<[^<]+?>', '', info)[:-1]
    else:
        return re.sub('<[^<]+?>', '', info)

def scrap_tambon2amphoe2changwat():
    w.set_lang('en')
    s = w.page(w.search('List of tambon in Thailand')[0])
    r = requests.get(s.url).text
    page = bs(r, 'html.parser')

    links = page.find_all('a')
    hrefs = [link.get('href') for link in links]
    urls = ['https://en.wikipedia.org' + h for h in hrefs if h is not None and 'List_of_tambon_in_Thailand_-_' in h]

    rows = []
    for url in tqdm(iterable=urls, desc=' urls', leave=False):
        r = requests.get(url).text
        page = bs(r, 'html.parser')
        table_classes = {"class": ["sortable", "plainrowheaders"]}
        wiki_table = page.findAll("table", table_classes)[0]

        for row in tqdm(iterable=wiki_table.findAll("tr"), desc=' wiki tables', leave=False):
            cells = row.findAll("td")
            if len(cells) != 0:
                tambon = remove_tags(str(cells[1]))
                amphoe = remove_tags(str(cells[3]))
                changwat = remove_tags(str(cells[5]))
                rows.append([tambon, amphoe, changwat])

    tambons = pd.DataFrame(data=rows, columns = ['Tambon', 'Amphoe', 'Changwat'])
    tambons.to_csv('address_extraction/resources/tambon2amphoe2changwat.csv', sep=',', index=False)

def scrap_khet2khwaengs():
    w.set_lang('en')
    s = w.page(w.search('Khwaeng')[0])
    r = requests.get(s.url).text
    page = bs(r, 'html.parser')
    table_classes = {"class": "wikitable"}
    wiki_table = page.findAll("table", table_classes)

    rows = []
    for tn in range(len(wiki_table)):
        khwaengs = []
        khet = ''
        for row in tqdm(iterable=wiki_table[tn].findAll("tr"), desc=' wiki tables', leave=False):
            cells = row.findAll("td")

            if len(cells) != 0 and 'Name' not in str(cells[2]):
                if 'rowspan' in str(cells[0]):
                    if khet != '':
                        rows.append((khet, khwaengs))
                    khwaengs = []
                    khet = remove_tags(str(cells[2]), remove_p=True)
                    khwaengs.append(remove_tags(str(cells[5]), remove_p=True))
                else:
                    khwaengs.append(remove_tags(str(cells[2]), remove_p=True))
    
    
        rows.append((khet, khwaengs))
    
    khwaengs = pd.DataFrame.from_records(data=rows, columns = ['Khet', 'Khwaeng'])
    khwaengs.to_csv('address_extraction/resources/khet2khwaengs.csv', sep=',', index=False)

if __name__ == '__main__':

    args = sys.argv[1]

    if args == 'tambon':
        scrap_tambon2amphoe2changwat()
    elif args == 'khwaeng':
        scrap_khet2khwaengs()
    else:
        raise NotImplementedError()
