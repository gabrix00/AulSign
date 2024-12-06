import os
import pandas as pd
from collections import Counter
import random
from sklearn.model_selection import train_test_split

# Funzione per ottenere le due parole più frequenti
def get_most_freq(lista):
    lista_cleaned = [segno.lower().strip() for segno in lista]
    frequency_count = Counter(lista_cleaned)
    top_two_words = frequency_count.most_common(2)

    if len(top_two_words) >= 2:
        return top_two_words[0][0] + '|' + top_two_words[1][0]
    elif len(top_two_words) == 1:
        return top_two_words[0][0]
    else:
        return ''

def save_to_txt(df, file_name):
    with open(file_name, "w", encoding='utf-8') as file:
        for _, row in df.iterrows():
            # Scrive annotato testo e relativo sign writing nel formato richiesto
            file.write(f"{row['annotated_texts_multilingual']}# {row['sign_writing']}\n")
    print(f"File {file_name} creato con successo.")


def combine_lines(path1: str, path2: str, output_file_path: str, shuffle: bool = False):
    # Read lines from both files
    with open(path1, "r") as lines_p1:
        lines_path1 = lines_p1.readlines()
    with open(path2, "r") as lines_p2:
        lines_path2 = lines_p2.readlines()
    
    # Combine lines from both files
    combined_lines = lines_path1 + lines_path2
    
    # Shuffle if requested
    if shuffle:
        random.seed(42)  
        random.shuffle(combined_lines)
    
    # Write to the output file
    with open(output_file_path, "w") as file:
        for line in combined_lines:
            file.write(line)
    
    print('Combined successfully test and dev sentences!')



def filtering(file:str,reduce_sentences=False):

    # Caricamento del dataset
    data_path = 'data'  # Sostituisci con il tuo percorso dei dati
    df = pd.read_csv(os.path.join(data_path, 'SignBank3.csv'))

    # Pulizia dei dati: rimozione di eventuali spazi e gestione dei valori mancanti
    df['spoken_language'] = df['spoken_language'].str.strip()
    df['sign_language'] = df['sign_language'].str.strip()
    df['Sentence/Sign'] = df['Sentence/Sign'].str.strip()
    df = df.dropna(subset=['spoken_language', 'sign_language', 'Sentence/Sign', 'annotated_texts_multilingual'])



    if file == 'corpus':
        # Filtro per 'en' (inglese) e 'ase' (lingua dei segni americana) per tipo 'Sentence'
        df_corpus = df.loc[(df['spoken_language'] == 'en') & 
                        (df['sign_language'] == 'ase') & 
                        (df['Sentence/Sign'] == 'Sign')]
        
         # Sostituisce il separatore '᛫' con '|' nella colonna 'annotated_texts_multilingual'
        df_corpus['annotated_texts_multilingual'] = df_corpus['annotated_texts_multilingual'].apply(
            lambda x: x.replace('᛫', '|') if isinstance(x, str) else '')
        

    if file == 'sentences':

        
        # Filtro per 'en' (inglese) e 'ase' (lingua dei segni americana) per tipo 'Sentence'
        df_corpus = df.loc[(df['spoken_language'] == 'en') & 
                        (df['sign_language'] == 'ase') & 
                        (df['Sentence/Sign'] == 'Sentence')]
        
         # Sostituisce il separatore '᛫' con '|' nella colonna 'annotated_texts_multilingual'
        df_corpus['annotated_texts_multilingual'] = df_corpus['annotated_texts_multilingual'].apply(
            lambda x: x.split('᛫')[0]if isinstance(x, str) else '')

    
        train_df, dev_test_df = train_test_split(df_corpus, test_size=0.05, random_state=42)

        # Seconda divisione: 5% dev e 5% test, dalla parte dev_test_df
        dev_df, test_df = train_test_split(dev_test_df, test_size=1/2, random_state=42)  
        if reduce_sentences:
            train_df = train_df.sample(n=int(df_corpus.shape[0]*0.05), random_state=42)  

        

        # Salvataggio nei file txt
        dir = os.path.join(data_path, f"preprocess_output_full")
        os.makedirs(os.path.dirname(dir), exist_ok=True)

        save_to_txt(train_df, os.path.join(data_path, f"preprocess_output_full/train_{file}.txt"))
        save_to_txt(dev_df, os.path.join(data_path, f"preprocess_output_full/dev_{file}.txt"))
        save_to_txt(test_df, os.path.join(data_path, f"preprocess_output_full/test_{file}.txt"))


    # Scrittura dei dati filtrati e processati nel file 'corpus.txt'
    output_file_path = os.path.join(data_path, f"{file}.txt")
    with open(output_file_path, "w", encoding='utf-8') as f:
        # Itera sulle righe del DataFrame filtrato
        for _, row in df_corpus.iterrows():
            # Ottiene le parole più frequenti nella colonna 'annotated_texts_multilingual'
            words = get_most_freq(row['annotated_texts_multilingual'].split('|'))
            # Scrive nel file in formato desiderato
            f.write(f"{words}# {row['sign_writing']}\n")

        if file =='corpus':
            # Caratteri di punteggiatura e simboli aggiunti
            f.write("unto|until# M534x515S26506502x485S10010519x485S1001a466x490\n")
            f.write("Full stop|.# M500x500S38800464x496\n")
            f.write("comma|pause# M500x500S38700463x496\n")
            f.write("?|question# M500x500S38900464x493\n")
            f.write("!|exclamation# M500x500S38810463x495\n")
            f.write("colen|:# M500x500S38a00464x490\n")

    print(f"File 'corpus.txt' creato con successo in {output_file_path}")
