import pandas as pd
import numpy as np
import os
from .clean import clean_symbol


" da usare per in get_result_pipeline.py (kaggle results)"

def precision(gold, pred):
    gold = set(gold)  # Remove duplicates
    pred = set(pred)  # Remove duplicates
    if len(pred) == 0:
        return 0  # Prevent division by zero when no predictions are made
    return len(pred & gold) / len(pred)

def recall(gold, pred):
    gold = set(gold)  # Remove duplicates
    pred = set(pred)  # Remove duplicates
    if len(gold) == 0:
        return 0  # Prevent division by zero if no actual gold labels exist
    return len(pred & gold) / len(gold)

def f1(recall, precision):
    if precision == 0 or recall == 0:
        return 0  # Return F1 score as 0 if either precision or recall is 0
    return 2 * (precision * recall) / (precision + recall)


def compute_f1(preds_path,gold_path,pred_glifi_output_path,gold_glifi_output_path,metrics_output_path):


    df_pred = pd.read_csv(preds_path, names=['symbol'])
    df_gold = pd.read_csv(gold_path, names=['symbol'])

    recall_list, precision_list, f1_list = [], [], []
    with open(pred_glifi_output_path, "w") as pred_glifi_file, \
        open(gold_glifi_output_path, "w") as gold_glifi_file, \
        open(metrics_output_path, "w") as metrics_file:
        # Intestazione del file delle metriche
        metrics_file.write("Row\tPrecision\tRecall\tF1\n")
        
        for index, row in df_pred.iterrows():
            pred_glifi = clean_symbol(row['symbol'], glue=' ', order=False)
            pred_glifi_file.write(pred_glifi + '\n')

            gold_glifi = clean_symbol(df_gold.loc[index,'symbol'], glue=' ',order=False)
            gold_glifi_file.write(gold_glifi + '\n')

            pred_glifi_list = pred_glifi.split()
            gold_glifi_list = gold_glifi.split()
            
            # Calcolo di precision, recall, e F1 per la riga corrente
            row_precision = precision(gold_glifi_list, pred_glifi_list)
            row_recall = recall(gold_glifi_list, pred_glifi_list)
            row_f1 = f1(row_recall, row_precision)

            # Aggiungi alla lista di ciascuna metrica
            precision_list.append(row_precision)
            recall_list.append(row_recall)
            f1_list.append(row_f1)

            # Salva i valori nel file delle metriche
            metrics_file.write(f"{index + 1}\t{row_precision:.4f}\t{row_recall:.4f}\t{row_f1:.4f}\n")
        
        metrics_file.write(f"Media F1: {np.mean(f1_list)}\n")

    # Output della media di F1 alla fine
    print("Media F1:", np.mean(f1_list))
