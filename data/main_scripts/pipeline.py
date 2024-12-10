import os
import pandas as pd
import warnings
import shutil
import logging

from typing import Optional, Union, List, Tuple

# Import local scripts (ensure these are in the correct relative path)
from scripts.pre_process import filtering, combine_lines
from scripts.split_corpus import file_creation 
from scripts.spm_encode import encode
from scripts.spm_pre_process import train_spm_process
from scripts.embedder import embedd_sentence_train, embedd_corpus
from scripts.unk_detector import unk_detector
from scripts.intermidiate_generation import dataset_gen

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Suppress warnings more selectively
warnings.filterwarnings("ignore", category=FutureWarning)

def save_to_txt(df: pd.DataFrame, file_name: str) -> None:
    """
    Save DataFrame to a text file with sentence and FSW.
    
    Args:
        df (pd.DataFrame): DataFrame to save
        file_name (str): Path to save the file
    """
    try:
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        with open(file_name, "w", encoding='utf-8') as file:
            for _, row in df.iterrows():
                file.write(f"{row['sentence']}# {row['fsw']}\n")
        logger.info(f"File {file_name} created successfully.")
    except IOError as e:
        logger.error(f"Error saving file {file_name}: {e}")

def copy_file_if_exists(
    source_repo: str, 
    target_repo: str, 
    file_name: str
) -> bool:
    """
    Copy a file from source to target repository if it exists.
    
    Args:
        source_repo (str): Source repository path
        target_repo (str): Target repository path
        file_name (str): Name of the file to copy
    
    Returns:
        bool: True if file was copied, False otherwise
    """
    source_path = os.path.join(source_repo, file_name)
    target_path = os.path.join(target_repo, file_name)

    try:
        if os.path.exists(source_path):
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            shutil.copy2(source_path, target_path)
            logger.info(f"File '{file_name}' copied from '{source_repo}' to '{target_repo}'.")
            return True
        else:
            logger.warning(f"File '{file_name}' not found in '{source_repo}'.")
            return False
    except (IOError, OSError) as e:
        logger.error(f"Error copying file: {e}")
        return False

def load_dataframe(
    file_path: str, 
    columns: List[str] = ["sentence", "fsw"]
) -> pd.DataFrame:
    """
    Load DataFrame from a file with error handling.
    
    Args:
        file_path (str): Path to the input file
        columns (List[str], optional): Column names. Defaults to ["sentence", "fsw"]
    
    Returns:
        pd.DataFrame: Loaded and cleaned DataFrame
    """
    try:
        df = pd.read_csv(file_path, sep="# ", names=columns,)
        df.drop_duplicates(inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df
    except Exception as e:
        logger.error(f"Error loading DataFrame from {file_path}: {e}")
        return pd.DataFrame(columns=columns)

def prepare_filtered_01(df,output_path,mode,model_name):
    df = df.sample(
        n=int(df.shape[0] * 0.05), 
        random_state=42)
    
    
    save_to_txt(df, f"{output_path}/train_filtered_0.1.txt")
    df.to_csv(f"{output_path}/file_comparison/train_filtered_0.1.csv",index=False)

    
    sentences_train_output_path = f"tools/sentences_train_embeddings_{mode}_01.json"
    print(f"create {sentences_train_output_path}")
    embedd_sentence_train(f"{output_path}/file_comparison/train_filtered_0.1.csv", model_name, sentences_train_output_path)

    return df

def process_data(
    vocabulary_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    mode: str = "full",
    model_name: Optional[str] = None
) -> None:
    """
    Comprehensive data processing pipeline with robust error handling.
    
    Args:
        vocabulary_path (Optional[str]): Path to vocabulary file
        output_dir (Optional[str]): Output directory path
        mode (str): Processing mode (full, filtered, filtered_01)
        model_name (Optional[str]): Embedding model name
    """
    # Validate inputs
    if not output_dir:
        raise ValueError("Output directory must be specified")
    
    logger.info(f"Starting data processing - Mode: {mode}")

    # Construct file paths
    output_subdir = f"{output_dir}/file_comparison"
    os.makedirs(output_subdir, exist_ok=True)

    train_data_path = f"{output_dir}/train_sentences.txt"
    dev_data_path = f"{output_dir}/dev_sentences.txt"
    test_data_path = f"{output_dir}/test_sentences.txt"

    # Initialize DataFrames
    df_train, df_dev, df_test = None, None, None

    # Data preprocessing based on mode
    try:
        if mode == "full":
            filtering(file="corpus")
            filtering(file="sentences", reduce_sentences=False)

            '''
            combine_lines(
                path1=train_data_path,
                path2="data/corpus.txt",
                output_file_path=f"{output_dir}/train_combined.txt",
                shuffle=True,
            )
            '''
            
            # Load datasets
            df_train = load_dataframe(train_data_path)
            df_dev = load_dataframe(dev_data_path)
            df_test = load_dataframe(test_data_path)

        elif mode in ["filtered", "filtered_01"]:

            # Copy dev and test files from full mode
            full_output_dir = output_dir.replace(mode, "full")
            copy_file_if_exists(full_output_dir, output_dir, "dev_sentences.txt")
            copy_file_if_exists(full_output_dir, output_dir, "test_sentences.txt")

            # Load dev and test datasets
            df_dev = load_dataframe(dev_data_path)
            df_test = load_dataframe(test_data_path)

        else:
            raise ValueError(f"Invalid processing mode: {mode}")

        
        if 'corpus_embeddings.json' not in os.listdir('tools'):
            embedd_corpus(vocabulary_path, model_name, 'tools/corpus_embeddings.json')

        if "test.csv" not in os.listdir(f"{output_dir}/file_comparison"):
            print(output_dir)
            print(os.listdir(f"{output_dir}/file_comparison"))
            #NB: il test set è sempre uguale non im porta avere full, filtered o filtered_01
            test_interm_output_path = f"{output_dir}/file_comparison/test_intermediate.csv"  
            test_df = dataset_gen(test_data_path, "tools/corpus_embeddings.json", test_interm_output_path)  #genero test_intermidiate (sentences) solo per avere la ground truth
            test_df = test_df.groupby(['sentence','fsw']).agg({'symbol': ' | '.join, 'word': ' # '.join}).reset_index()
            test_df.to_csv(f"data/preprocess_output_{mode}/file_comparison/test.csv",index=False)

        # Intermediate dataset generation and UNK detection
        #if df_train is not None and vocabulary_path:

        if mode in ["full","filtered"]:
            interm_output_path = f"{output_subdir}/train_intermediate_{mode}.csv"   
            if f"train_intermediate_{mode}.csv" not in os.listdir(output_subdir):
                print(os.listdir(output_subdir))
                print(f"{interm_output_path} not found! It will be created...")

                if mode == "full":
                    df_intermidate = dataset_gen(train_data_path, 'tools/corpus_embeddings.json', interm_output_path)
                    unk_detector(
                        interm_output_path, 
                        threshold=None, 
                        train=True, 
                        vocab_path='tools/corpus_embeddings.json'
                    )

                    #salvalo anche su filter dato che intermidiate non filtrato è lo stesso per entrambi
                    if os.makedirs("data/preprocess_output_filtered/file_comparison", exist_ok=True):
                        df_intermidate.to_csv("data/preprocess_output_filtered/file_comparison/train_intermediate_filtered.csv",index=False) 
                    
                    df_intermidate = df_intermidate.groupby(['sentence', 'fsw']).agg({'symbol': ' | '.join, 'word': ' # '.join}).reset_index()
                    
                    
            else:
                print(f"{interm_output_path} found! It will be loaded...")
                df_intermidate = pd.read_csv(interm_output_path)
                df_intermidate = df_intermidate.groupby(['sentence', 'fsw']).agg({'symbol': ' | '.join, 'word': ' # '.join}).reset_index()
                
            if mode == "filtered":
                df_intermidate = df_intermidate[~df_intermidate['word'].str.contains('<unk>')]
                save_to_txt(df_intermidate, "data/preprocess_output_filtered/train_filtered.txt")
            
                if f"sentences_train_embeddings_{mode}_01.json" not in os.listdir('tools'):
                    print("save files for case filtered_01")
                    prepare_filtered_01(df_intermidate,"data/preprocess_output_filtered_01",mode,model_name)
                
            # Save processed train data
            train_output_file = f"{output_subdir}/train_{mode}.csv"
            df_intermidate.to_csv(train_output_file, index=False)
            logger.info(f"Train {mode} saved to: {train_output_file}")

            # Sentence embedding
            if model_name:
                sentences_train_output_path = f"tools/sentences_train_embeddings_{mode}.json"
                if f"sentences_train_embeddings_{mode}.json" not in os.listdir('tools'):
                    embedd_sentence_train(train_output_file, model_name, sentences_train_output_path)
                else:
                    print(f"sentences_train_embeddings_{mode} is already present!")

        if mode == "filtered":
            df_train = load_dataframe(f"data/preprocess_output_filtered/train_filtered.txt")

        elif mode == "filtered_01":
            df_train = load_dataframe(f"data/preprocess_output_filtered_01/train_filtered_0.1.txt")


        # Vocabulary and file creation
        vocab = load_dataframe(vocabulary_path, columns=["word", "fsw"])

        # Create various output files for train, dev, and test sets
        for case, df in [("train", df_train), ("dev", df_dev), ("test", df_test)]:
            if df is not None:
                output_prefix = f"{output_subdir}/{case}"
                file_creation(
                    df,
                    f"{output_prefix}_fsw.txt",
                    f"{output_prefix}_symbol.txt",
                    f"{output_prefix}_ordered_symbol.txt",
                    f"{output_prefix}_factor_x.txt",
                    f"{output_prefix}_factor_y.txt",
                    f"{output_prefix}_ordered_factor_x.txt",
                    f"{output_prefix}_ordered_factor_y.txt",
                    f"{output_prefix}_sentences.txt",
                    vocab=vocab if case == "train" else None,
                )

        # SentencePiece processing
        train_spm_process(f"{output_subdir}/train_sentences.txt",output_subdir)

        # Encoding
        model_path = f"{output_subdir}/spm.model"
        for case in ["train", "dev", "test"]:
            encode(
                f"{output_subdir}/{case}_sentences.txt",
                f"{output_subdir}/{case}_sentences.spm.txt",
                model_path,
            )

        logger.info(f"Process completed for mode: {mode}")

    except Exception as e:
        logger.error(f"Error in data processing: {e}")
        raise

def main():
    """Main execution function."""
    model_name = "mixedbread-ai/mxbai-embed-large-v1"

    processing_configs = [
        {
            "vocabulary_path": "data/corpus.txt",
            "output_dir": "data/preprocess_output_full",
            "mode": "full",
            "model_name": model_name
        },
        {
            "vocabulary_path": "data/corpus.txt",
            "output_dir": "data/preprocess_output_filtered",
            "mode": "filtered",
            "model_name": model_name
        },
        {
            "vocabulary_path": "data/corpus.txt",
            "output_dir": "data/preprocess_output_filtered_01",
            "mode": "filtered_01",
            "model_name": model_name
        }
    ]

    for mode in ["full","filtered","filtered_01"]:
        output_subdir = f"data/preprocess_output_{mode}/file_comparison"
        print(output_subdir)
        os.makedirs(output_subdir, exist_ok=True)

    for config in processing_configs:
        try:
            process_data(**config)
        except Exception as e:
            logger.error(f"Failed to process configuration: {config}")
            logger.error(f"Error details: {e}")

if __name__ == "__main__":
    main()