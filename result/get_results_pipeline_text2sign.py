import pandas as pd
import sys
import os
import argparse

# Aggiungere dinamicamente "data/main_scripts" a sys.path
# Aggiungere la directory principale del progetto a sys.path
current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, '..', 'data', 'main_scripts'))
sys.path.append(project_root)

# Verifica del percorso aggiunto
#print(f"Added to sys.path: {project_root}")

# Importare i moduli
from scripts.post_process_factor import file_creation
from scripts.factors_evaluate import calculate_distances
from scripts.sacreblue_metrics import calculate_metrics, evaluate_per_line
from scripts.compute_f1 import compute_f1

def main(path):

    #path = 'result/text2sign_2024_12_06_12_17'

    #preds = f"{path}/predictions.txt"
    splitted_path = path.split('/')
    path , file_name = '/'.join(splitted_path[:-1]), ''.join(splitted_path[-1])
    #print(path) debug
    #print(file_name) debug
    
    df = pd.read_csv(f"{path}/{file_name}")
    df_pred = df[['pred_fsw_seq']]
    df_pred.rename(columns={'pred_fsw_seq':'fsw'},inplace=True)
    df_gold = df[['gold_fsw_seq']]
    df_gold.rename(columns={'gold_fsw_seq':'fsw'},inplace=True)


    gold_fsw_output_path = f"{path}/gold_fsw.txt"
    gold_factor_x_output_path = f"{path}/test_factor_x.txt"
    gold_factor_y_output_path = f"{path}/test_factor_x.txt"
    gold_symbol_output_path =  f"{path}/test_symbol.txt"
    gold_symbol_ordered_output_path = f"{path}/test_ordered_symbol.txt"
    gold_factor_x_ordered_output_path = f"{path}/test_ordered_factor_x.txt"
    gold_factor_y_ordered_output_path = f"{path}/test_ordered_factor_y.txt"
    gold_glifi_output_path = f"{path}/test_glifi.txt"




    predictions_fsw_output_path = f"{path}/predictions_fsw.txt"
    predictions_symbol_output_path = f"{path}/predictions_symbol.txt"
    predictions_symbol_ordered_output_path = f"{path}/predictions_ordered_symbol.txt"
    predictions_factor_x_output_path = f"{path}/predictions_factor_x.txt"
    predictions_factor_y_output_path = f"{path}/predictions_factor_y.txt"
    predictions_factor_x_ordered_output_path = f"{path}/predictions_ordered_factor_x.txt"
    predictions_factor_y_ordered_output_path = f"{path}/predictions_ordered_factor_y.txt"
    predictions_glifi_output_path = f"{path}/predictions_glifi.txt"
    

    file_creation(df_pred, predictions_fsw_output_path, predictions_symbol_output_path, predictions_symbol_ordered_output_path, 
                  predictions_factor_x_output_path, predictions_factor_y_output_path,
                  predictions_factor_x_ordered_output_path, predictions_factor_y_ordered_output_path)

    file_creation(df_gold, gold_fsw_output_path, gold_symbol_output_path, gold_symbol_ordered_output_path, 
                  gold_factor_x_output_path, gold_factor_y_output_path,
                  gold_factor_x_ordered_output_path, gold_factor_y_ordered_output_path)


    calculate_distances(predictions_factor_x_output_path, gold_factor_x_output_path, predictions_factor_y_output_path, gold_factor_y_output_path,output_path=path+'/factor_metrics.txt')

    calculate_distances(predictions_factor_x_ordered_output_path, gold_factor_x_ordered_output_path, predictions_factor_y_ordered_output_path, gold_factor_y_ordered_output_path, output_path=path+'/factor_metrics.txt', order_factor=True )


    calculate_metrics(predictions_symbol_output_path, gold_symbol_output_path, output_path=path+'/metrics.txt')

    calculate_metrics(predictions_symbol_ordered_output_path, gold_symbol_ordered_output_path, output_path=path+'/ordered_metrics.txt')

    evaluate_per_line(predictions_symbol_output_path, gold_symbol_output_path, output_path=path+'/evaluate_per_line.txt')
    
    compute_f1(predictions_symbol_output_path, gold_symbol_output_path, predictions_glifi_output_path, gold_glifi_output_path, metrics_output_path= path+'/metrics_f1.txt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True)

    args = parser.parse_args()

    if args.result:
        main(path=args.result)
    else:
        print("Error: Probably result_path is wrong")
#main('result/text2sign_2024_12_06_12_17/result_2024_12_06_12_17.csv')