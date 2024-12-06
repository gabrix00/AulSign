
import pandas as pd
import re
from tqdm import tqdm
import json
from collections import Counter
from scripts.clean import clean_sign, clean_symbol


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
    

    

def dataset_gen(sentences_path:str, vocab_path:str,outuput_path:str):# fsw_path:str,


    df = pd.read_csv(sentences_path, sep='# ',names=['sentence','fsw'],engine='python')
    df.drop_duplicates(inplace=True)
    
    df['symbol'] = df['fsw'].apply(lambda x: clean_sign(x.split())[1])
    df['symbol'] = df['symbol'].apply(lambda x: clean_symbol(x,glue='|',order=False))

    

    with open(vocab_path, 'r') as file:
        content = file.read()
        vocab = json.loads(content)

    list_sentence = []
    list_word = []
    list_symbol = []
    list_goldfsw = []

    pbar = tqdm(total=df.shape[0], desc="Processing Sentence")

    
    for _, row in df.iterrows():
        sentence, fsw, symbols = row['sentence'], row['fsw'], row['symbol']
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
                list_symbol.append(symbol) #appendo il simbolo originale non il temp_ordered_symbol
            else:
                list_word.append("<unk>")
                list_symbol.append(symbol)
            

            list_sentence.append(sentence)
            list_goldfsw.append(fsw)
        
        
        pbar.update(1)
    pbar.close()
    
    #print(len(list_sentence)) #debug
    #print(len(list_word)) #debug
    #print(len(list_sign)) #debug

    df = pd.DataFrame({
        'sentence': list_sentence,  
        'word': list_word,
        'symbol': list_symbol,
        'fsw': list_goldfsw
    })

    df.to_csv(outuput_path,index=False)
   
    return df
