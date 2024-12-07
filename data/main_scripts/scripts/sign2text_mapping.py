
import pandas as pd
import json
from collections import Counter

import sys
import os

current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, 'data', 'main_scripts', 'scripts'))
sys.path.append(project_root)

from clean import clean_sign, clean_symbol

def get_most_freq(lista):
    lista_cleaned = [item.lower().strip() for item in lista]
    frequency_count = Counter(lista_cleaned)
    top_two_words = frequency_count.most_common(2)

    if len(top_two_words) >= 2:
        return top_two_words[0][0] + '|' + top_two_words[1][0]
    elif len(top_two_words) == 1:
        return top_two_words[0][0]
    else:
        return ''

def sign2text(fsw_seq:str, vocab_path:str):

    df = pd.DataFrame({'fsw': [fsw_seq]})

    df['symbol'] = df['fsw'].apply(lambda x: clean_sign(x.split())[1])
    df['symbol'] = df['symbol'].apply(lambda x: clean_symbol(x,glue='|',order=False))

    #print('\n') #deubg
    #print(df.loc[0,'fsw']) #deubg
    #print('\n')  #deubg
    #print('\n') #deubg
    #print(df.loc[0,'symbol']) #deubg


    with open(vocab_path, 'r') as file:
        content = file.read()
        vocab = json.loads(content)

    list_word = []

    fsw, symbols = df.loc[0,'fsw'], df.loc[0,'symbol']
    for symbol in symbols.split('|'):

        #print(symbol) #debug
        temp_ordered_symbol = ' '.join(sorted(symbol.split()))
        #print(temp_ordered_symbol) #debug

        mapped_words = [
            ', '.join(entry['word'])
            for entry in vocab
            if 'symbol' in entry
            and any(temp_ordered_symbol == s for s in (entry['symbol'] if isinstance(entry['symbol'], list) else [entry['symbol']]))
        ]

        if mapped_words:
            canonical = get_most_freq(' '.join(mapped_words).split(','))
            list_word.append(canonical)
        else:
            list_word.append("<unk>")
        
   
    return ' # '.join(list_word)