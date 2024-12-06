from scripts.clean import clean_sign,clean_symbol
import os
import pandas as pd


def custom_replace(s):
    parts = s.split('|')
    if len(parts) == 3:
        return f"{parts[0]}{parts[1]}x{parts[2]}" #symbol-factor1-factor2
    return s 

def file_creation(df, fsw_output_path, symbol_output_path, symbol_ordered_output_path,
                  factor_x_output_path, factor_y_output_path, factor_x_ordered_output_path,
                  factor_y_ordered_output_path,factors=False):

    with open(fsw_output_path, "w") as fsw_file, \
         open(symbol_output_path, "w") as symbol_file, \
         open(symbol_ordered_output_path, "w") as symbol_ordered_file, \
         open(factor_x_output_path, "w") as factor_x_file, \
         open(factor_y_output_path, "w") as factor_y_file, \
         open(factor_x_ordered_output_path, "w") as factor_x_ordered_file, \
         open(factor_y_ordered_output_path, "w") as factor_y_ordered_file:
        for index, row in df.iterrows():

            if factors:
                row = ' '.join([custom_replace(el) for el in row['fsw'].split()])
                row = row.replace(' S','S')
            else:
                row = ' '.join([el for el in row['fsw'].split()])

            
            #print(row)
            fsw_file.write(f"{row}\n")
            _, symbols, factor_x, factor_y  = clean_sign(row.split(' '))
            symbol_file.write(f"{symbols}\n")
            factor_x_file.write(f"{factor_x}\n")  
            factor_y_file.write(f"{factor_y}\n") 



            _, symbols_ord, factor_x_ord, factor_y_ord  = clean_sign(row.split(' '),sort=True)
            symbol_ordered_file.write(f"{symbols_ord}\n")
            factor_x_ordered_file.write(f"{factor_x_ord}\n") 
            factor_y_ordered_file.write(f"{factor_y_ord}\n") 

