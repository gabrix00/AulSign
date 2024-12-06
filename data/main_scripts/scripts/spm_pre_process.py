import sentencepiece as spm

def train_spm_process(input_path,vocab_size=2000):
    model_prefix = '/'.join(input_path.split('/')[:3])
    #print(model_prefix)
    spm.SentencePieceTrainer.train(
        input=input_path,
        #model_prefix = "spm",
        model_prefix=f"{model_prefix}/spm",
        vocab_size=vocab_size,
        model_type='bpe',
        character_coverage=1.0
    )
