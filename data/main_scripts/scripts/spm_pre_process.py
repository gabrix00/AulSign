import sentencepiece as spm
from pathlib import Path

def train_spm_process(input_path, output_subdir, vocab_size=2000):
    # Usa pathlib per gestire i percorsi
    input_path = Path(input_path)
    output_subdir = Path(output_subdir)
    
    # Verifica se la directory di output esiste, altrimenti creala
    output_subdir.mkdir(parents=True, exist_ok=True)

    # Prepara il percorso per il modello (senza aggiungere "spm" come sottocartella)
    model_prefix = output_subdir / "spm"  # Questo salverà il modello come spm.model direttamente in output_subdir
    
    # Esegui il training del modello
    spm.SentencePieceTrainer.train(
        input=str(input_path),  # SentencePiece richiede una stringa, quindi converto il Path in stringa
        model_prefix=str(model_prefix),  # Salva il modello direttamente in output_subdir/spm
        vocab_size=vocab_size,
        model_type='bpe',
        character_coverage=1.0
    )
