import pandas as pd
import sys
import os

sys.path.append(os.path.abspath("/Users/gabrieletuccio/Developer/"))
from data.main_scripts.scripts.sacreblue_metrics import evaluate_per_line

if __name__ == '__main__':

    path = 'result/sign2text_2024_12_02_12_20'

    df = pd.read_csv(f"{path}/result_2024_12_02_12_20.csv")

    # Percorsi di output per i due file di testo
    predictions_file_output_path = os.path.join(path, 'predictions.txt')
    gold_file_output_path = os.path.join(path, 'gold.txt')

    df['pseudo_sentence'].to_csv(predictions_file_output_path, index=False, header=False)
    df['sentence'].to_csv(gold_file_output_path, index=False, header=False)

    evaluate_per_line(predictions_file_output_path, gold_file_output_path, output_path=path+'/evaluate_per_line.txt')
    