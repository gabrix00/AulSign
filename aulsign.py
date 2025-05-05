import os
import json
import numpy as np
import pandas as pd
import logging
from collections import Counter
from sentence_transformers import SentenceTransformer
import warnings
from datetime import datetime
from sklearn.preprocessing import normalize
import requests
import json
import argparse
from openai import OpenAI

import sys
import os
current_dir = os.path.dirname(__file__)
#print(f"current dir is: {current_dir}")
project_root = os.path.abspath(os.path.join(current_dir, 'data', 'main_scripts','scripts'))
#print(f"project root is: {project_root}")
sys.path.append(project_root)

from sign2text_mapping import sign2text


client = OpenAI(
  organization=OPENAI_ORGANIZATION,
  project=OPENAI_PROJECT,
  api_key=OPENAI_API_KEY
)

warnings.filterwarnings("ignore", category=FutureWarning)
# Set up logging configuration
logging.basicConfig(
    filename='AulSign.log',  # Log to a file
    level=logging.DEBUG,         # Log everything, including debug info
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
    filemode='w'                 # Overwrite the log file each run
)

'''
client = OpenAI(
  organization=os.getenv("OPENAI_ORGANIZATION"),
  project=os.getenv("OPENAI_PROJECT"),
  api_key=os.getenv("OPENAI_API_KEY")
)
'''

def query_ollama(messages, model="mistral:latest"):
    logging.info(f"Querying model: {model}")
    url = "http://localhost:11434/api/chat"

    options = {"seed": 42,"temperature": 0.1}


    payload = {
        "model": model,
        "messages": messages,
        "options": options,
        "stream": False
    }

    response = requests.post(url, json=payload)

    if response.status_code == 200:
        return response.json()["message"]["content"]
    else:
        return f"Error: {response.status_code}, {response.text}"

def check_repetition(text, threshold=0.2):
    if not text:
        return False
    
    words = [word.strip for word in text.split('#')]

    unique_words = len(set(words))
    total_words = len(words)

    if "<unk>" in words:
        logging.debug(f"Check repetition: '<unk>' was generated in the answer")
        return True

    
    is_repetitive = unique_words < total_words * threshold
    logging.debug(f"Check repetition: {is_repetitive} (Unique: {unique_words}, Total: {total_words})")
    return is_repetitive


# Function to merge predictions with gold data and compute metrics
def prepare_dataset(prediction: pd.DataFrame, validation: pd.DataFrame, modality:str):
    if modality=='text2sign':
        validation = validation.rename(columns={'fsw':'gold_fsw_seq','symbol': 'gold_symbol_seq', 'word': 'gold_cd'}) 
        metrics = prediction.merge(validation[['gold_symbol_seq','gold_cd', 'sentence','gold_fsw_seq']], on=['sentence'])
    elif modality=='sign2text':
        validation = validation.rename(columns={'word': 'gold_cd'}) 
        metrics = prediction.merge(validation[['sentence','gold_cd']], on=['gold_cd'])
    return metrics

# Define cosine similarity function if it's missing
def cos_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def find_most_similar_sentence(user_embedding, train_sentences: pd.DataFrame, n=3, unk_threshold=7):
    # Estrai gli embedding, le decomposizioni e le frasi dal DataFrame
    sentence_embeddings = np.vstack(train_sentences["embedding_sentence"].values)  # Matrix of sentence embeddings
    decompositions = train_sentences["decomposition"].values
    sentences = train_sentences["sentence"].values
    
    # Normalizza gli embedding delle frasi e l'embedding utente
    sentence_embeddings = normalize(sentence_embeddings, axis=1)
    user_embedding = normalize(user_embedding.reshape(1, -1), axis=1)
    
    # Calcola le similarità usando un'unica moltiplicazione matrice-vettore
    similarities = np.dot(sentence_embeddings, user_embedding.T).flatten()  # Shape (num_sentences,)
    
    # Imposta la similarità a zero per le frasi con troppi "<unk>"
    unk_counts = np.array([d.count("<unk>") for d in decompositions])
    similarities[unk_counts > unk_threshold] = 0  # Penalizza le frasi con troppi "<unk>"
    
    # Ottieni gli indici delle top-n frasi più simili
    top_n_indices = np.argsort(similarities)[-n:][::-1]
    
    # Ritorna le decomposizioni e le frasi corrispondenti alle top-n similitudini
    return [decompositions[i] for i in top_n_indices], [sentences[i] for i in top_n_indices]


def find_most_similar_canonical_entry(user_embedding, vocabulary: pd.DataFrame, n=30):
    # Extract embeddings and words from the vocabulary
    vocabulary_embeddings = np.vstack(vocabulary["embedding"].values)  # Matrix of embeddings
    vocabulary_words = vocabulary["word"].values
    
    # Normalize vocabulary embeddings and user embedding
    vocabulary_embeddings = normalize(vocabulary_embeddings, axis=1)
    user_embedding = normalize(user_embedding.reshape(1, -1), axis=1)
    
    # Compute cosine similarities for all entries in one matrix multiplication
    similarities = np.dot(vocabulary_embeddings, user_embedding.T).flatten()  # Shape (vocabulary_size,)
    
    # Get a sorted list of indices based on similarity scores
    sorted_indices = np.argsort(similarities)[::-1]  # Sort in descending order
    
    # Initialize lists for canonical entries and similarities
    canonical_list = []
    canonical_similarities = []
    
    for idx in sorted_indices:
        if len(canonical_list) >= n:  # Stop once we have n entries
            break
        
        # Get canonical entry for the current word
        canonical_entry = get_most_freq(vocabulary_words[idx])
        
        # Check for duplicates in canonical entries
        if canonical_entry not in canonical_list:
            canonical_list.append(canonical_entry)
            canonical_similarities.append(similarities[idx])
    
    # Return the top n canonical entries and their similarities
    return canonical_list#, canonical_similarities


def get_most_freq(lista:list):
    lista_cleaned = []
    for segno in lista:
        segno_pulito = segno.lower().strip()
        if segno_pulito not in lista_cleaned:
            lista_cleaned.append(segno_pulito)

    frequency_count = Counter(lista_cleaned)
    #print(frequency_count)
    top_two_words = frequency_count.most_common(2)

    if len(top_two_words) >= 2:
        first_word = top_two_words[0][0]
        second_word = top_two_words[1][0]

        return first_word+'|'+second_word
    else:
        first_word = top_two_words[0][0]
        return first_word

def get_most_freq_fsw(lista_fsw):
    if isinstance(lista_fsw,str):
        return lista_fsw
    else:
        frequency_count = Counter(lista_fsw)
        max_freq_word = frequency_count.most_common(1)[0][0]
        return max_freq_word


def get_fsw_exact(vocabulary: pd.DataFrame, can_desc_answer, model, top_k=10):
    # Extract vocabulary embeddings and words
    vocabulary_embeddings = np.vstack(vocabulary["embedding"].values)  # Create a matrix of all embeddings
    vocabulary_words = vocabulary["word"].values
    vocabulary_fsw = vocabulary["fsw"].values

    # Normalize vocabulary embeddings for cosine similarity
    vocabulary_embeddings = normalize(vocabulary_embeddings, axis=1)

    fsw_seq = []
    can_desc_association_seq = []
    joint_prob = 1

    for can_d in can_desc_answer:
        # Encode the candidate description and normalize
        can_d_emb = model.encode(can_d, normalize_embeddings=True).reshape(1, -1)  # Shape (1, embedding_dim)

        # Compute cosine similarities using matrix multiplication
        similarities = np.dot(vocabulary_embeddings, can_d_emb.T).flatten()  # Shape (vocabulary_size,)

        # Get the indices of the top_k most similar elements
        top_k_indices = np.argsort(similarities)[-top_k:][::-1]  # Indices of top-k elements
        top_k_words = vocabulary_words[top_k_indices]
        top_k_fsws = vocabulary_fsw[top_k_indices]
        top_k_similarities = similarities[top_k_indices]

        # Check for an exact match in the top_k elements
        exact_match_index = next((i for i, word in enumerate(top_k_words) if get_most_freq(word) == can_d.strip()), None)

        if exact_match_index is not None:
            # Exact match found
            most_similar_word = get_most_freq(top_k_words[exact_match_index])
            fsw = top_k_fsws[exact_match_index]
            max_similarity = 1  # Assign maximum similarity for an exact match
        else:
            # If no exact match, use the most similar word semantically
            max_index = 0  # First element in the sorted top_k (highest similarity)
            most_similar_word = get_most_freq(top_k_words[max_index])
            fsw = top_k_fsws[max_index]
            max_similarity = top_k_similarities[max_index]

        # Append the result
        logging.info(fsw)
        fsw_seq.append(get_most_freq_fsw(fsw))  # Append to fsw sequence
        joint_prob *= max_similarity  # Multiply joint probability
        can_desc_association_seq.append(most_similar_word)

        # Logging
        logging.debug(f"Word: {can_d}")
        logging.debug(f"Most similar word in vocabulary: {most_similar_word}")
        logging.debug(f"Similarity: {max_similarity}")
        logging.debug(f"Fsw_seq: {' '.join(fsw_seq)}")
        logging.debug("---")

    # Compute geometric mean of joint probability
    joint_prob = pow(joint_prob, 1 / len(can_desc_association_seq))
    
    return ' '.join(fsw_seq), ' # '.join(can_desc_association_seq), np.round(joint_prob, 3)


# Process input sentence through retrieval-augmented generation (RAG)
def AulSign(input:str, rules_prompt_path:str, train_sentences:pd.DataFrame, vocabulary:pd.DataFrame, model, ollama:bool, modality:str):
    """
AulSign: A function for translating between text and Formal SignWriting (FSW) or vice versa.

This function leverages embeddings, similarity matching, and language models to facilitate
translations based on the specified modality (`text2sign` or `sign2text`).

Args:
    input (str): 
        The sentence or sign sequence to be analyzed and translated.
    rules_prompt_path (str): 
        Path to a file containing predefined prompts and rules to guide the language model.
    train_sentences (pd.DataFrame): 
        A dataset containing sentences and their embeddings for training or similarity matching.
    vocabulary (pd.DataFrame): 
        A table of vocabulary entries with canonical descriptions and embeddings, used for matching.
    model: 
        The embedding model used to convert sentences or sign sequences into vector representations.
    ollama (bool): 
        Specifies whether to use the `query_ollama` method for querying the language model.
    modality (str): 
        The translation mode:
        - `'text2sign'`: Converts text to Formal SignWriting sequences.
        - `'sign2text'`: Converts Formal SignWriting to textual sentences.

Returns:
    For `modality == "text2sign"`:
        tuple:
            - answer (str): 
                The translated text or decomposition provided by the language model.
            - fsw (list): 
                A list of Formal SignWriting sequences associated with the translation.
            - can_desc_association_seq (list): 
                A list of canonical descriptions associated with the FSW sequences.
            - joint_prob (float): 
                The joint probability of the most likely translation path.

    For `modality == "sign2text"`:
        str: 
            The reconstructed textual sentence translated from the input sign sequence.

    If an invalid modality is provided:
        str: 
            Returns 'error' to indicate invalid input.

Raises:
    Exception: 
        Logs and raises errors encountered during API calls or message construction.
    """
   
    sent_embedding = model.encode(input, normalize_embeddings=True)

    if modality =='text2sign':
    
        similar_canonical = find_most_similar_canonical_entry(sent_embedding, vocabulary, n=100)
        #print(similar_canonical)

        
        similar_canonical_str = ' # '.join(similar_canonical)

        # Load the rules prompt from the file
        with open(rules_prompt_path, 'r') as file:
            rules_prompt = file.read().format(similar_canonical=similar_canonical_str)

        # Find the most similar sentences from training set
        decomposition, sentences = find_most_similar_sentence(
            user_embedding=sent_embedding, 
            train_sentences=train_sentences, 
            n=20
        )

        messages = [{"role": "system", "content": rules_prompt}]
        for sentence, decomposition in zip(sentences, decomposition):
            # Ensure each message has 'role' and 'content' keys
            if sentence and decomposition:
                messages.append({"role": "user", "content": sentence})
                messages.append({"role": "assistant", "content": decomposition})#.replace(' | ',' # ')})
            else:
                logging.warning("Missing 'sentence' or 'decomposition' in messages.")

        messages.append({"role": "user", "content": "decompose the following sentence as shown in the previous examples"})
        messages.append({"role": "user", "content": input})
        
        # Validate the constructed messages before converting to prompt text
        valid_messages = []
        for message in messages:
            if 'role' in message and 'content' in message:
                valid_messages.append(message)
                logging.debug(message)
            else:
                logging.error(f"Invalid message format detected: {message}")

        if ollama:
            # Query the LLM using query_ollama instead of llm_pipeline
            answer = query_ollama(messages)#, model="mistral:7b-instruct-fp16")

            logging.info("\n[LOG] Ollama model Answer:")
            logging.info(answer)

            can_description_answer = answer.split('#')
        else:
            try:
                # Initial API call
                completion = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    temperature=0
                )
                answer = completion.choices[0].message.content

                if check_repetition(answer):
                # Optional: Repetition check
                    presence_penalty = 0.6
                    completion = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=messages,
                        presence_penalty=presence_penalty,
                        temperature=0
                    )
                    logging.info(f"presence_penalty: {presence_penalty}")
                    answer = completion.choices[0].message.content
                    logging.info('ANSWER: GPT')
                    logging.info(answer + '\n\n')

                    # Update parsed answer
                    can_description_answer = answer.split('#')
                    
                else:
                    logging.info('ANSWER: GPT')
                    logging.info(answer + '\n\n')

                    # Split for further processing
                    can_description_answer = answer.split('#')


            except Exception as e:
                logging.error(f"Error during GPT API call: {e}")

        # Map canonical descriptions to most similar words in vocabulary
        fsw, can_desc_association_seq, joint_prob = get_fsw_exact(
            vocabulary=vocabulary, 
            can_desc_answer=can_description_answer, 
            model=model
        )

        return answer, fsw, can_desc_association_seq, joint_prob
    
    elif modality =='sign2text':

       # Load the rules prompt from the file
        with open(rules_prompt_path, 'r') as file:
            rules_prompt = file.read()


        # Find the most similar sentences from training set
        decomposition, sentences = find_most_similar_sentence(
            user_embedding=sent_embedding, 
            train_sentences=train_sentences, 
            n=30
        )

        messages = [{"role": "system", "content": rules_prompt}]
        for sentence, decomposition in zip(sentences, decomposition):
            # Ensure each message has 'role' and 'content' keys
            if sentence and decomposition:
                messages.append({"role": "user", "content": decomposition})
                messages.append({"role": "assistant", "content": sentence}) # qui stiamo invertendo il task! dalla decomposition vogliamo che l'assistant ci dia la sentence
            else:
                logging.warning("Missing 'sentence' or 'decomposition' in messages.")

        messages.append({"role": "user", "content": "reconstruct the sentence as shown on the examples above"})
        messages.append({"role": "user", "content": input})
        
        # Validate the constructed messages before converting to prompt text
        valid_messages = []
        for message in messages:
            if 'role' in message and 'content' in message:
                valid_messages.append(message)
                logging.debug(message)
            else:
                logging.error(f"Invalid message format detected: {message}")

        if ollama:
            # Query the LLM using query_ollama instead of llm_pipeline
            answer = query_ollama(messages)#, model="mistral:7b-instruct-fp16")

            logging.info("\n[LOG] Ollama model Answer:")
            logging.info(answer)

            can_description_answer = answer.split('#')
        else:
            try:
                # Initial API call
                completion = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    temperature=0
                )
                answer = completion.choices[0].message.content
                logging.info('ANSWER: GPT')
                logging.info(answer + '\n\n')


            except Exception as e:
                logging.error(f"Error during GPT API call: {e}")

        return answer

    else:
        return 'error'
    

def main(modality, setup=None, input=None):
    np.random.seed(42)
    current_time = datetime.now().strftime("%Y_%m_%d_%H_%M")
    #data_path = f"data/preprocess_output_{setup}/file_comparison"
    data_path = "data/preprocess_output_filtered_01/file_comparison" #il test set è sempre lo stesso ne basta uno!
    corpus_embeddings_path = 'tools/corpus_embeddings.json'
    if setup is None:
        print(f"Running in inference mode!")
        sentences_train_embeddings_path = f"tools/sentences_train_embeddings_filtered_01.json"
    else:
        print(f"Running in experimental mode with setup: {setup}!")
        logging.info(f"Running in experimental mode with setup: {setup}!")
        sentences_train_embeddings_path = f"tools/sentences_train_embeddings_{setup}.json"
        #sentences_train_embeddings_path =  f"data/additional/sentences_train_embeddings_{setup}.json"

    rules_prompt_path_text2sign = 'tools/rules_prompt_text2sign.txt'
    rules_prompt_path_sign2text = 'tools/rules_prompt_sign2text.txt'

    # Model to use for sentence embeddings
    model_name = "mixedbread-ai/mxbai-embed-large-v1"
    model = SentenceTransformer(model_name)

    # Load embeddings
    with open(corpus_embeddings_path, 'r') as file:
        corpus_embeddings = pd.DataFrame(json.load(file))

    with open(sentences_train_embeddings_path, 'r') as file:
        sentences_train_embeddings = pd.DataFrame(json.load(file))

    if input:  # Se è fornita una frase personalizzata
        if modality == 'text2sign':
            answer, fsw_seq, can_desc_association_seq, joint_prob = AulSign(
                input=input,
                rules_prompt_path=rules_prompt_path_text2sign,
                train_sentences=sentences_train_embeddings,
                vocabulary=corpus_embeddings,
                model=model,
                ollama=False,
                modality=modality
            )
            #print(f"Input Sentence: {input}")
            print(f"Canonical Descriptions: {can_desc_association_seq}")
            print(f"Translation (FSW): {fsw_seq}")
            #print(f"Canonical Descriptions: {can_desc_association_seq}")
            #print(f"Joint Probability: {joint_prob}")
        
        elif modality == 'sign2text': #qui l'input è una FSW seq, che deve essere mappata in canonicals
            mapped_input = sign2text(input,corpus_embeddings_path)

            answer = AulSign(
                input=mapped_input,
                rules_prompt_path=rules_prompt_path_sign2text,
                train_sentences=sentences_train_embeddings,
                vocabulary=corpus_embeddings,
                model=model,
                ollama=False,
                modality=modality
            )
            print(f"Input Sign Voucaboualry Mapping: {input}")
            print(f"Translation (Text): {answer}")

    else:  # Flusso standard con testset
        test_path = os.path.join(data_path, f"test.csv")
        test = pd.read_csv(test_path)
        #test = test.head(10)

        if modality == 'text2sign':
            list_sentence = []
            list_answer = []
            list_fsw_seq = []
            can_desc_association_list = []
            prob_of_association_list = []

            for index, row in test.iterrows():
                sentence = row['sentence']
                answer, fsw_seq, can_desc_association_seq, joint_prob = AulSign(
                    input=sentence,
                    rules_prompt_path=rules_prompt_path_text2sign,
                    train_sentences=sentences_train_embeddings,
                    vocabulary=corpus_embeddings,
                    model=model,
                    ollama=False,
                    modality=modality
                )

                list_sentence.append(sentence)
                list_answer.append(answer)
                list_fsw_seq.append(fsw_seq)
                can_desc_association_list.append(can_desc_association_seq)
                prob_of_association_list.append(joint_prob)
            
            df_pred = pd.DataFrame({
                'sentence': list_sentence,
                'pseudo_cd': list_answer,
                'pred_cd': can_desc_association_list,
                'joint_prob': prob_of_association_list,
                'pred_fsw_seq': list_fsw_seq
            })
            output_path = os.path.join('result', f"{modality}_{current_time}")
            os.makedirs(output_path, exist_ok=True)
            df_pred = prepare_dataset(df_pred,test,modality)
            df_pred.to_csv(os.path.join(output_path, f'result_{current_time}.csv'), index=False)

        elif modality == 'sign2text':

            list_answer = []
            list_gold_cd = []

            for index, row in test.iterrows():
                dec_sentence = row['word']
                answer = AulSign(
                    input=dec_sentence,
                    rules_prompt_path=rules_prompt_path_sign2text,
                    train_sentences=sentences_train_embeddings,
                    vocabulary=corpus_embeddings,
                    model=model,
                    #ollama=False,
                    ollama=True,
                    modality=modality
                )
                list_gold_cd.append(dec_sentence)
                list_answer.append(answer)
            
            df_pred = pd.DataFrame({
                'pseudo_sentence': list_answer,
                'gold_cd': list_gold_cd,
            })
            output_path = os.path.join('result', f"{modality}_{current_time}")
            os.makedirs(output_path, exist_ok=True)
            df_pred = prepare_dataset(df_pred,test,modality)
            df_pred.to_csv(os.path.join(output_path, f'result_{current_time}.csv'), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, help="Mode of operation: text2sign or sign2text")
    parser.add_argument("--setup", help="Experimental setup configuration")
    parser.add_argument("--input", help="Input text or sign sequence")

    args = parser.parse_args()

    # Ensure either setup or input is provided, but not both mandatory
    if args.setup and not args.input:
        main(modality=args.mode, setup=args.setup)
    elif args.input and not args.setup:
        main(modality=args.mode, input=args.input)
    elif args.setup and args.input:
        print("Error: Both setup and input cannot be provided simultaneously.")
    else:
        print("Error: Either setup or input must be provided.")