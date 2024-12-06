import re
from signwriting.formats.fsw_to_sign import fsw_to_sign

def clean_sign(fsw_list: list, glue=' ', sort = False):
    list_factor_x, list_factor_y = [], []
    list_symbols = []

    for item in fsw_list:
        sign = fsw_to_sign(item)  

        # Aggiungi il simbolo principale con posizione se esiste
        if 'box' in sign and 'symbol' in sign['box'] and 'position' in sign['box']:
            #head del simbolo
            box_symbol = sign['box']['symbol']
            box_factor_x = sign['box']['position'][0]
            box_factor_y = sign['box']['position'][1]
            list_symbols.append(str(box_symbol))
            list_factor_x.append(str(box_factor_x))
            list_factor_y.append(str(box_factor_y))
            if sort:
            # Sort symbols alphabetically by 'symbol' key
                symbols = sorted(sign['symbols'], key=lambda el: el['symbol'])
            else:
                symbols = sign['symbols']

            for el in symbols:
                symbol = el['symbol']
                factor_x = el['position'][0]
                factor_y = el['position'][1]
                list_factor_x.append(str(factor_x))
                list_factor_y.append(str(factor_y))
                list_symbols.append(symbol)

    return glue.join(fsw_list), glue.join(list_symbols), glue.join(list_factor_x), glue.join(list_factor_y)



def replace_match(match):
    return "|" + match.group(0)  # Helper function to add '|' separator

def clean_symbol(symbols, pattern=r'\b(M|L|R|B)\b', glue=' ', order=False, add_box_symbol=False):
    # Aggiungo i separatori ai simboli tramite il pattern
    symbols = re.sub(pattern, replace_match, symbols)
   
    
    # Rimuovo il primo carattere '|' se presente
    if symbols.startswith('|'):
        symbols = symbols[1:]
   

    symbols_cleaned_list = []
    
    # Itero sui simboli separati dal carattere '|'
    for symbol in symbols.split('|'):
        # Controllo se il simbolo inizia con uno dei caratteri target (M, L, R, B)
      
        if symbol.startswith(('M', 'L', 'R', 'B')):
            box_symbol = symbol[0]  # Assegno la lettera come box_symbol
            symbol_pure = symbol[1:].strip()  # Rimuovo la lettera dal simbolo
     
        if order:
            symbol_pure = ' '.join(sorted(symbol_pure.split()))

        else:
            symbol_pure = ' '.join(symbol_pure.split())

        # Aggiungo box_symbol al simbolo epurato
        if add_box_symbol:
            symbols_cleaned_list.append(box_symbol + ' ' + symbol_pure)
        else:
            symbols_cleaned_list.append(symbol_pure)
        
            
    # Unisco tutti i simboli epurati con `glue`
    return glue.join(symbols_cleaned_list)

        