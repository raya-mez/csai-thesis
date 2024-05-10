import re
import bz2
import pickle
import logging
import numpy as np
from gensim import corpora
from collections import Counter
from gensim.models import LsiModel

# Default input and output files
wordids_path = "data/wiki_wordids.txt.bz2"
bow_corpus_path = "data/wiki_bow.mm"
vocab_path = "data/vocab.pkl"
lsa_model_path = "models/wiki_lsi_model.model"

def create_vocabulary(wordids_path=wordids_path, bow_corpus_path=bow_corpus_path, output_file_path=vocab_path):
    """
    Create a vocabulary with the 5000 most frequent word IDs in the BoW corpus.

    The function reads a word ID mapping file and a BoW corpus file, aggregates word
    frequencies across the corpus, and creates a vocabulary dictionary containing the
    5000 most frequent words as values and their IDs as keys. The vocabulary dictionary 
    is then saved to the specified output file.

    Parameters:
        wordids_path (str): Path to the word ID mapping file.
        bow_corpus_path (str): Path to the BoW corpus file.
        output_file_path (str): Path to the output file for saving the vocabulary.

    Returns:
        None: The function saves the vocabulary dictionary to the specified output file.
    """
    # Create a dictionary mapping words to ids from the word ID file
    id_word_dict = {}
    with bz2.open(wordids_path, 'rt') as f:
        for line in f: # Line format: wordID word doc_frequency
            match = re.match(r'(\d+)\s+(\S+)', line) 
            if match:
                word_id = int(match.group(1))
                word = match.group(2)
                id_word_dict[word_id] = word
    logging.info(f"Loaded id-word dictionary with {len(id_word_dict)} entries.")
    
    # Initialize a dictionary to store word frequencies
    total_word_freqs = Counter()

    # Load the BoW corpus
    logging.info("Loading BoW corpus...")
    bow_corpus = corpora.MmCorpus(bow_corpus_path)

    # Aggregate word type frequencies across the entire corpus
    logging.info("Aggregating word frequencies...")
    for document in bow_corpus:
        for word_id, freq in document:
            total_word_freqs[word_id] += freq

    # Get 5000 most frequent word IDs
    most_freq_ids = [item[0] for item in total_word_freqs.most_common(5000)]
    
    # Create the vocabulary dictionary with the 5000 most frequent words
    vocab = {id: id_word_dict[id] for id in most_freq_ids}
    logging.info(f"Created vocabulary with {len(vocab)} entries.")

    # Save the vocabulary dictionary to the output file using pickle
    with open(output_file_path, 'wb') as f:
        pickle.dump(vocab, f)
    logging.info(f"Vocabulary saved to {output_file_path}")


def load_vocabulary(file_path=vocab_path):
    """
    Load the pickled vocabulary from the specified file path. 
    The vocabulary is a dictionary containing words as values 
    and their corresponding word IDs as keys.
    
    Parameters:
        file_path (str): The path to the file containing the vocabulary dictionary. Defaults to "data/vocab.pkl".
        
    Returns:
        dict: A dictionary containing words as values and their corresponding word IDs as keys.
    """
    # Open the file in binary read mode
    with open(file_path, 'rb') as f:
        # Load the vocabulary dictionary using pickle
        vocabulary = pickle.load(f)
    
    # Return the vocabulary dictionary
    return vocabulary

def filter_vocab_by_wordlength(vocab, word_lengths=[3,4,5,6,7]):
    filtered_vocab = {}
    for id, word in vocab.items():
        if len(word) in word_lengths:
            filtered_vocab[id] = word
    return filtered_vocab


def word_ids_by_word_length(vocab):
    words_ids_by_length = {}
    for length in range(3,8):
        ids = [id for id, word in vocab.items() if len(word) == length] 
        words_ids_by_length[length] = ids
    return words_ids_by_length


def get_vocabulary_embeddings_dict(vocab, lsa_model=lsa_model_path): # vocab_path
    """
    Extract the embeddings of the words in the vocabulary.
    
    The function loads a vocabulary dictionary and the LSA model. 
    It then retrieves the embeddings of the words in the vocabulary
    from the `projection.u` attribute of the LSI model.
    
    Args:
        lsa_model (str): The path to the LSI model file. Defaults to "models/wiki_lsi_model.model".
        vocab (dict): The dictionary where keys are word IDs and values are the corresponding words.
        
    Returns:
        dict: A dictionary mapping word IDs in the vocabulary to their embeddings.
    """
    
    # Load the LSI model from the specified path
    lsa_model = LsiModel.load(lsa_model_path)    
    
    # Create a dictionary mapping words to their embeddings (the left singular vectors from the LSI model)
    embeddings_dict = {id:lsa_model.projection.u[id] for id in vocab.keys()}
    
    # Return the dictionary of words and their embeddings
    return embeddings_dict