import pandas as pd
import sys
import os
import argparse

# Aggiungere dinamicamente la directory radice del progetto
current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, '..', 'data', 'main_scripts'))
sys.path.append(project_root)

# Verifica del percorso aggiunto
#print(f"Added to sys.path: {project_root}") debug

from scripts.sacreblue_metrics import evaluate_per_line

def main(path):

    #path = 'result/sign2text_2024_12_02_12_20'
    splitted_path = path.split('/')
    path , file_name = '/'.join(splitted_path[:-1]), ''.join(splitted_path[-1])
    #print(path) debug
    #print(file_name) debug

    df = pd.read_csv(f"{path}/{file_name}")

    # Percorsi di output per i due file di testo
    predictions_file_output_path = os.path.join(path, 'predictions.txt')
    gold_file_output_path = os.path.join(path, 'gold.txt')

    df['pseudo_sentence'].to_csv(predictions_file_output_path, index=False, header=False)
    df['sentence'].to_csv(gold_file_output_path, index=False, header=False)

    evaluate_per_line(predictions_file_output_path, gold_file_output_path, output_path=path+'/evaluate_per_line.txt')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True)

    args = parser.parse_args()

    if args.result:
        main(path=args.result)
    else:
        print("Error: Probably result_path is wrong")