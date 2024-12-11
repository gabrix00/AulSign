import pandas as pd
import sys
import os

current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, '..', 'data', 'main_scripts'))
sys.path.append(project_root)

# Verifica del percorso aggiunto
print(f"Added to sys.path: {project_root}")

from scripts.post_process_factor import file_creation
from scripts.factors_evaluate import calculate_distances
from scripts.sacreblue_metrics import calculate_metrics, evaluate_per_line
from scripts.compute_f1 import compute_f1

def main(path):

    #NB full is specified but the same is for filtered and filtered_01 since test and dev sets are always the same
    gold_factor_x_path = 'data/preprocess_output_full/file_comparison/test_factor_x.txt'
    gold_factor_y_path = 'data/preprocess_output_full/file_comparison/test_factor_x.txt'
    gold_symbol_path =  'data/preprocess_output_full/file_comparison/test_symbol.txt'
    gold_symbol_ordered_path = 'data/preprocess_output_full/file_comparison/test_ordered_symbol.txt'
    gold_factor_x_ordered_path = 'data/preprocess_output_full/file_comparison/test_ordered_factor_x.txt'
    gold_factor_y_ordered_path = 'data/preprocess_output_full/file_comparison/test_ordered_factor_y.txt'
    gold_glifi_path = "data/preprocess_output_full/file_comparison/test_glifi.txt"

    #path = 'jiang_results'

    splitted_path = path.split('/')
    path = '/'.join(splitted_path[:-1])
    print(path)

    preds = f"{path}/predictions.txt"
    df = pd.read_csv(preds, names=['fsw'])
    

    predictions_fsw_output_path = f"{path}/predictions_fsw.txt"
    predictions_symbol_output_path = f"{path}/predictions_symbol.txt"
    predictions_symbol_ordered_output_path = f"{path}/predictions_ordered_symbol.txt"
    predictions_factor_x_output_path = f"{path}/predictions_factor_x.txt"
    predictions_factor_y_output_path = f"{path}/predictions_factor_y.txt"
    predictions_factor_x_ordered_output_path = f"{path}/predictions_ordered_factor_x.txt"
    predictions_factor_y_ordered_output_path = f"{path}/predictions_ordered_factor_y.txt"
    predictions_glifi_output_path = f"{path}/predictions_glifi.txt"
    

    

    file_creation(df, predictions_fsw_output_path, predictions_symbol_output_path, predictions_symbol_ordered_output_path,
                  predictions_factor_x_output_path, predictions_factor_y_output_path,
                  predictions_factor_x_ordered_output_path, predictions_factor_y_ordered_output_path,factors=True)

    calculate_distances(predictions_factor_x_output_path, gold_factor_x_path, predictions_factor_y_output_path, gold_factor_y_path, output_path=path+'/factor_metrics.txt')

    calculate_distances(predictions_factor_x_ordered_output_path, gold_factor_x_ordered_path, predictions_factor_y_ordered_output_path, gold_factor_y_ordered_path, output_path=path+'/factor_metrics.txt', order_factor=True )

    calculate_metrics(predictions_symbol_output_path, gold_symbol_path, output_path=path+'/metrics.txt')

    calculate_metrics(predictions_symbol_ordered_output_path, gold_symbol_ordered_path, output_path=path+'/ordered_metrics.txt')

    evaluate_per_line(predictions_symbol_output_path, gold_symbol_path, output_path=path+'/evaluate_per_line.txt')

    compute_f1(predictions_symbol_output_path, gold_symbol_path, predictions_glifi_output_path, gold_glifi_path, metrics_output_path= path+'/metrics_f1.txt')

#main('jiang_results/asl-95_full_result/predictions.txt')
