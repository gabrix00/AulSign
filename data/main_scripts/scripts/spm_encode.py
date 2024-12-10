import sentencepiece as spm

def encode(input_file, output_file, model_path):
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    
    # Aggiungi 'encoding="utf-8"' per supportare tutti i caratteri Unicode
    with open(input_file, 'r', encoding='utf-8') as fin, open(output_file, 'w', encoding='utf-8') as fout:
        for line in fin:
            # Codifica ogni riga e scrivi nel file di output
            fout.write(' '.join(sp.encode(line.strip(), out_type=str)) + '\n')
