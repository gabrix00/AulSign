import numpy as np
import pandas as pd
import json
import os
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

def embedd_sentence_train(sentences_train_path, model_name: str, output_path: str):
    df = pd.read_csv(sentences_train_path)
    df.rename(columns={'word': 'decomposition'}, inplace=True)
    model = SentenceTransformer(model_name)

    emb_list = []
    pbar = tqdm(total=df.shape[0], desc="Processing Sentences")
    for index, row in df.iterrows():
        sentence = row['sentence']
        decomposition = row['decomposition']
        embedd = model.encode(sentence)
        emb_list.append({'id': index, 'sentence': sentence, 'decomposition': decomposition, 'embedding_sentence': embedd.tolist()})
        pbar.update(1)
    pbar.close()

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(emb_list, f, ensure_ascii=False, indent=4)

def save_to_txt(df, filepath):
    df.to_csv(filepath, index=False, sep=' ', header=False)

def prepare_filtered_01(df, output_path, mode, model_name, seed):
    sampled_df = df.sample(n=int(df.shape[0] * 0.05), random_state=seed)
    
    os.makedirs(f"{output_path}/file_comparison", exist_ok=True)
    txt_path = f"{output_path}/train_filtered_0.1_seed{seed}.txt"
    csv_path = f"{output_path}/file_comparison/train_filtered_0.1_seed{seed}.csv"
    
    save_to_txt(sampled_df, txt_path)
    sampled_df.to_csv(csv_path, index=False)
    
    sentences_train_output_path = f"{output_path}/sentences_train_embeddings_{mode}_01_seed{seed}.json"
    print(f"Creating {sentences_train_output_path}")
    embedd_sentence_train(csv_path, model_name, sentences_train_output_path)

output_path = "data/additional"
train_output_file = f"{output_path}/train_filtered.csv"
model_name = "sentence-transformers/all-MiniLM-L6-v2"
df = pd.read_csv(train_output_file)

'''
for i in range(9):
    seed = np.random.randint(0, 100000)
    print(seed)
    prepare_filtered_01(df, output_path, "filtered", model_name, seed)
'''
# Lista di seed già definita
seeds = [10350,11397,11617,14694,16251,40691,64969,82924,87237]


for seed in seeds:
    print(seed)
    prepare_filtered_01(df, output_path, "filtered", model_name, seed)