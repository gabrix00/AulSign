import subprocess

def calculate_metrics(predictions_path, gold_path, output_path):
    """
    Esegue il comando sacrebleu per calcolare i punteggi BLEU e chrF tra un file di predizioni e un file di riferimento.

    Args:
        predictions_path (str): Il percorso del file di predizioni.
        gold_path (str): Il percorso del file di gold standard.
        output_path (str): Il percorso del file di output dove salvare i risultati.
    """
    try:
        # Definisce il comando da eseguire
        command = [
            "sacrebleu", gold_path,
            "-i", predictions_path,
            "-m", "bleu", "chrf",
            "-w 2",
            "--chrf-lowercase",
            "--score-only"
        ]
        #-i viene usata per specificare il file delle predizioni,
        
        # Esegue il comando e salva l'output nel file specificato
        with open(output_path, "w") as output_file:
            subprocess.run(command, stdout=output_file, check=True)
        
        print(f"Calcolo delle metriche completato. Risultati salvati in {output_path}.")
        
    except subprocess.CalledProcessError as e:
        print(f"Errore nell'esecuzione del comando: {e}")



from sacrebleu import metrics


def evaluate_per_line(predictions_path, gold_path, output_path):
    """
    Calcola BLEU e chrF sia per ogni riga sia considerando l'intero corpus.

    Args:
        predictions_path (str): Il percorso del file con le predizioni.
        gold_path (str): Il percorso del file con i riferimenti.
        output_path (str): Il percorso dove salvare i risultati.
    """
    try:
        bleu = metrics.BLEU(effective_order=True)
        chrf = metrics.CHRF()

        sentence_bleu_scores = []
        sentence_chrf_scores = []

        # Leggi tutte le linee
        with open(predictions_path, "r", encoding="utf-8") as pred_file, \
             open(gold_path, "r", encoding="utf-8") as gold_file, \
             open(output_path, "w", encoding="utf-8") as output_file:
            
            pred_lines = [line.strip() for line in pred_file]
            gold_lines = [line.strip() for line in gold_file]

            # Controlla se il numero di righe corrisponde
            if len(pred_lines) != len(gold_lines):
                raise ValueError("Il numero di righe in predizioni e gold non corrisponde.")
            
            # Valutazione per riga
            for idx, (pred, gold) in enumerate(zip(pred_lines, gold_lines), start=0):
                if not pred or not gold:
                    output_file.write(f"Linea {idx}: Riga vuota o inconsistente\n")
                    continue
                
                bleu_score = bleu.sentence_score(pred, [gold])
                chrf_score = chrf.sentence_score(pred, [gold])

                sentence_bleu_scores.append(bleu_score.score)
                sentence_chrf_scores.append(chrf_score.score)

                # Scrivi i risultati per ogni riga
                output_file.write(f"Linea {idx}:\n")
                output_file.write(f"Gold: {gold}\n")
                output_file.write(f"Pred: {pred}\n")
                output_file.write(f"BLEU: {bleu_score.score:.2f}\n")
                output_file.write(f"chrF: {chrf_score.score:.2f}\n\n")
            
            # Valutazione su tutto il corpus
            corpus_bleu_score = bleu.corpus_score(pred_lines, [gold_lines]).score
            corpus_chrf_score = chrf.corpus_score(pred_lines, [gold_lines]).score

            # Scrivi i risultati del corpus
            output_file.write("Risultati Corpus:\n")
            output_file.write(f"Corpus BLEU: {corpus_bleu_score:.2f}\n")
            output_file.write(f"Corpus chrF: {corpus_chrf_score:.2f}\n")
            
        print(f"Calcolo completato. Risultati salvati in {output_path}.")
    
    except Exception as e:
        print(f"Errore durante il calcolo delle metriche: {e}")
        

