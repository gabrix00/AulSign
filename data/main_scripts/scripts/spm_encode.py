import sentencepiece as spm

# Funzione per tokenizzare e salvare in file di output
def encode(input_file, output_file, model_path):
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
        for line in fin:
            fout.write(' '.join(sp.encode(line.strip(), out_type=str)) + '\n')