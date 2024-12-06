import pandas as pd
import numpy as np
import json 
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import warnings
from scripts.clean import clean_sign, clean_symbol
warnings.filterwarnings("ignore", category=FutureWarning)



def embedd_corpus(corpus_train_path:str,model_name:str,output_path:str): 
    df = pd.read_csv(corpus_train_path,sep='# ',names=['word','fsw'],engine='python')
    df.drop_duplicates(inplace=True)
    
    df['symbol'] = df['fsw'].apply(lambda x: clean_sign([x])[1])
    df['symbol'] = df['symbol'].apply(lambda x: clean_symbol(x,order=True))
    df_grouped = df.groupby('symbol').agg({'fsw': ', '.join, 'word': '|'.join}).reset_index()
    
    model = SentenceTransformer(model_name)
    emb_list = []
    pbar = tqdm(total=df_grouped.shape[0], desc="Processing Vocabs Words")


    for index, row in df_grouped.iterrows():
        word = row['word']
        embedd = model.encode(word)
        emb_list.append({'id': index, 'symbol': row['symbol'], 'word':[word for word in row['word'].split('|')], 'fsw':[sign for sign in row['fsw'].split(', ')], 'embedding': embedd.tolist()})  # Convert the embedding to a list
        pbar.update(1)
    pbar.close()

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(emb_list, f, ensure_ascii=False, indent=4)




def embedd_sentence_train(sentences_train_path,model_name:str,output_path:str):
    df = pd.read_csv(sentences_train_path)
    df.rename(columns={'word':'decomposition'},inplace=True)
    model = SentenceTransformer(model_name)
                                
    emb_list = []
    pbar = tqdm(total=df.shape[0], desc="Processing Sentences")
    for index, row in df.iterrows():
        sentence = row['sentence']
        decomposition = row['decomposition']
        embedd = model.encode(sentence)
        emb_list.append({'id': index, 'sentence':sentence, 'decomposition':decomposition, 'embedding_sentence': embedd.tolist()})  # Convert the embedding to a list
        
        pbar.update(1)
    pbar.close()

  
    with open(output_path, 'w',encoding='utf-8') as f:
        json.dump(emb_list, f, ensure_ascii=False, indent=4)

