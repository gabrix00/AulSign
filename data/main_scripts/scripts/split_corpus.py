import pandas as pd
import os
from scripts.clean import clean_sign


def file_creation(df, fsw_output_path, symbol_output_path, symbol_ordered_output_path, factor_x_output_path, factor_y_output_path, factor_x_ordered_output_path, factor_y_ordered_output_path, sentence_output_path, vocab=None):
    """
    Scrive i dati del dataframe in tre file di output.
    """
    with open(fsw_output_path, "w") as fsw_file, \
         open(symbol_output_path, "w") as symbol_file, \
         open(symbol_ordered_output_path, "w") as symbol_ordered_file, \
         open(factor_x_output_path, "w") as factor_x_file, \
         open(factor_y_output_path, "w") as factor_y_file, \
         open(factor_x_ordered_output_path, "w") as factor_x_ordered_file, \
         open(factor_y_ordered_output_path, "w") as factor_y_ordered_file, \
         open(sentence_output_path, "w") as sentence_file:
        for _, row in df.iterrows():
            fsw, symbols, factors_x, factors_y  = clean_sign(row['fsw'].split())
            fsw_file.write(f"{fsw}\n")
            symbol_file.write(f"{symbols}\n")
            #symbol_ordered_file.write(f"{clean_symbol(symbols,order=True,add_box_symbol=True)}\n") 
            factor_x_file.write(f"{factors_x}\n") 
            factor_y_file.write(f"{factors_y}\n")  
            sentence_file.write(f"{row['sentence']}\n") 

            _, symbols_ord, factor_x_ord, factor_y_ord  = clean_sign(row['fsw'].split(' '),sort=True)
            symbol_ordered_file.write(f"{symbols_ord}\n")
            factor_x_ordered_file.write(f"{factor_x_ord}\n") 
            factor_y_ordered_file.write(f"{factor_y_ord}\n") 



        if vocab is not None:
            for _, row in vocab.iterrows():
                if 'fsw' in row and 'word' in row:
                #    if row['word'] != 'afferrare':
                    
                    fsw, symbols, factors_x, factors_y  = clean_sign([row['fsw']])
                    fsw_file.write(f"{fsw}\n")
                    symbol_file.write(f"{symbols}\n")
                    #symbol_ordered_file.write(f"{clean_symbol(symbols,order=True,add_box_symbol=True)}\n") 
                    factor_x_file.write(f"{factors_x}\n")  
                    factor_y_file.write(f"{factors_y}\n") 
                    sentence_file.write(f"{row['word']}\n") 

                    _, symbols_ord, factor_x_ord, factor_y_ord  = clean_sign([row['fsw']],sort=True)
                    symbol_ordered_file.write(f"{symbols_ord}\n")
                    factor_x_ordered_file.write(f"{factor_x_ord}\n") 
                    factor_y_ordered_file.write(f"{factor_y_ord}\n") 


