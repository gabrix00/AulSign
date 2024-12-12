import pandas as pd
import json
import os
from sentence_transformers import SentenceTransformer


def symbol2pseudofsw(symbol:str):
    pseudofsw = 'M500x500'+symbol.replace(' ','500x500')
    pseudofsw +='500x500'
    return pseudofsw
    
def update_json_vocab_with_embeddings(rename_mapping, vocab_path, model_name="mixedbread-ai/mxbai-embed-large-v1"):
    """
    Update a JSON vocabulary file with unknown token mappings and embeddings based on a given threshold.
    
    Parameters:
    rename_mapping (dict) of most common unk according training set
    vocab_path (str): Path to the JSON vocabulary file.
    model_name (str, optional): The name of the SentenceTransformer model to use for embeddings. Defaults to "mixedbread-ai/mxbai-embed-large-v1".
    """
    model = SentenceTransformer(model_name)

    try:
        with open(vocab_path, 'r') as file:
            vocab = json.load(file)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")

    
    # Add the new tokens and their embeddings
    for k, v in rename_mapping.items():
        dim = len(vocab)-1 #perchè parte da id = 0
        vocab.append({
            'id': dim+1,
            'symbol': k,
            'word': [v],
            'fsw': [symbol2pseudofsw(k)],
            'embedding': model.encode(v).tolist()
        })
    
    # Save the updated vocabulary to the JSON file
    with open(vocab_path, 'w',encoding='utf-8') as file:
        json.dump(vocab, file,  ensure_ascii=False, indent=4)



def unk_detector(df_path:str, threshold, train=False, vocab_path=None):
    df = pd.read_csv(df_path)
    
    "unk detector save  in a csv file the frequencies of unknown signs and return a list of more freq sign based on a setted treshold"
    "unk detector call < update_json_vocab_with_embeddings > if vocaboulary path is passed, in order to update that vocab withthe unk detected "

    filtered_df = df[df['word'] == '<unk>']
    count_per_sign = filtered_df.groupby('symbol').size()
    count_per_sign = count_per_sign.sort_values(ascending=False)

    os.makedirs("eda", exist_ok=True)

    if train:
        count_per_sign.to_csv('eda/count_per_symbol_train.csv', header=True)
    else:
        count_per_sign.to_csv('eda/count_per_symbol_val.csv', header=True)


    if threshold != None:
        filtered_count_per_sign = count_per_sign.head(threshold)
        rename_mapping = {sign: f'unk_{i+1}' for i, sign in enumerate(filtered_count_per_sign.index)}
        df['word'] = df['symbol'].map(rename_mapping).fillna(df['word'])

        if vocab_path != None:
            update_json_vocab_with_embeddings(rename_mapping,vocab_path)

    return df
