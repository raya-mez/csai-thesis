import sys
import bz2
import random
import pickle as pkl
import logging
from gensim import corpora
from collections import Counter


# ---------- Configurations ----------
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s : %(levelname)s : %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])

if len(sys.argv) < 8:
    logging.error("Usage: python build_vocab_inspect.py <wordids_path> <bow_corpus_path> <vocab_path> <selected_words_path>")
    sys.exit(1)

wordids_path = sys.argv[1] # "models/wiki_wordids.txt.bz2"
bow_corpus_path = sys.argv[2] # "models/wiki_tfidf.mm"
vocab_path = sys.argv[3] # "data/vocab.pkl"
selected_words_path = sys.argv[4] # "data/200_words.txt"
celex_path = sys.argv[5] # "data/celex_dict.pkl"
controlled_vocab_path = sys.argv[6] # "data/vocab_controlled.pkl"
controlled_selected_words_path = sys.argv[7] # "data/200_words_controlled.txt"


# ---------- Get word-id mappings ----------
# Create a dictionary mapping IDs to words
id2word_dict = {}

with bz2.open(wordids_path, 'rt', encoding='utf-8') as f:
    first_line = True
    for line in f:
        if first_line:
            first_line = False
            continue  # Skip the first line - it contains the number of documents in the corpus
        # Split the line into components
        id, word, _ = line.split() # NOTE: the last element in the line is the word's document frequency
        id2word_dict[id] = word

logging.info(f"Loaded id2word dictionary with {len(id2word_dict)} entries.")
logging.info(f"First 10 id2word items: {list(id2word_dict.items())[:10]}")


# ---------- Get overall word frequencies ----------
# NOTE: the wordids file contains the document frequencies of the words (i.e., the number of documents they appear in).
# However, we want to get their overall frequencies in the corpus.

# Initialize a dictionary to store word frequencies
total_word_freqs = Counter()

# Load the BoW corpus containing word IDs and their frequencies in each document
logging.info("Loading BoW corpus...")
bow_corpus = corpora.MmCorpus(bow_corpus_path)

# Aggregate word type frequencies across the entire corpus
logging.info("Aggregating word frequencies...")
for document in bow_corpus:
    for word_id, freq in document:
        total_word_freqs[word_id] += freq


# ---------- Create a vocabulary with the 5,000 most frequent word IDs (no filtering) ----------
logging.info("Creating vocabulary with the 5,000 most frequent word IDs...")

# Sort the word frequency dictionary by values (frequencies)
sorted_word_freqs = total_word_freqs.most_common() # yields a list of tuples

# Create a dictionary containing the 5000 most frequent words keyed by their IDs
most_freq_ids_5000 = []
counter = 0
for id, _ in sorted_word_freqs:
    if counter == 5000:
        break
    # Add the id-word pair to the vocabulary if it is part of the 10,000 filtered words collected in id2word_dict
    if id in id2word_dict:
        most_freq_ids_5000.append(id)
        counter += 1
vocab = {id:id2word_dict[id] for id in most_freq_ids_5000}

# Store the vocabulary in a pickle file for easy access
with open(vocab_path, 'wb') as f:
    pkl.dump(vocab, f)
logging.info(f"Saved non-filtered vocabulary to {vocab_path}")


### Repeat, controlling for morphological status of words in the vocabulary ###
# ---------- Load the CELEX dictionary ----------
with open(celex_path, 'rb') as f:
    celex_dict = pkl.load(f)
logging.info(f"Loaded CELEX dictionary with {len(celex_dict)} entries")

# ---------- Function to check morphological status ----------
def is_monomorphemic_lemma(word_id, id2word_dict, celex_dict):
    """
    Check if a word ID corresponds to a monomorphemic lemma according to CELEX.
    Return False if it does not, as well as if the word ID is not found in the 
    dictionary mapping 10,000 IDs to words or if the word is not found in CELEX.
    """
    if word_id not in id2word_dict:
        return False
    word = id2word_dict[word_id]
    if word not in celex_dict:
        return False
    return celex_dict[word]['morphstatus'] == 'M' and celex_dict[word]['lemma'] == celex_dict[word]['worddia']

# # Extract all wordforms which are monomorphemic lemmas from the CELEX dictionary
# celex_monomorph_lemmas = []
# for item in celex_dict:
#     if celex_dict[item]['morphstatus'] == 'M' and celex_dict[item]['lemma'] == celex_dict[item]['worddia']:
#         celex_monomorph_lemmas.append(celex_dict[item]['worddia'])
# logging.info(f"Extracted {len(celex_monomorph_lemmas)} monomorphemic lemmas from CELEX")


# ---------- Create a vocabulary with the 5000 most frequent monomorphemic wordforms which are lemmas ----------
logging.info("Creating controlled vocabulary...")

# Create a dictionary containing the 5000 most frequent monomorphemic words keyed by their IDs
most_freq_monomorph_ids_5000 = []
counter = 0
for id, _ in sorted_word_freqs:
    if counter == 5000:
        break
    # Check if the word is monomorphemic according to CELEX
    if is_monomorphemic_lemma(id, id2word_dict, celex_dict):
    # if id in id2word_dict and id2word_dict[id] in celex_monomorph_lemmas:
        most_freq_monomorph_ids_5000.append(id)
        counter += 1

controlled_vocab = {id:id2word_dict[id] for id in most_freq_monomorph_ids_5000}

# Store the vocabulary in a pickle file for easy access
with open(controlled_vocab_path, 'wb') as f:
    pkl.dump(controlled_vocab, f)
logging.info(f"Saved filtered vocabulary to {controlled_vocab_path}")


# ---------- Select 200 word types for inspection from each vocabulary ----------
def select_word_types_for_inspection(sorted_word_ids, id2word_dict, output_file):
    """
    Select 200 word types for inspection and write them to a file.
    
    Args:
        sorted_word_ids (list): List of word IDs to select from, sorted by frequency (from most frequent to least frequent).
        id2word_dict (dict): Dictionary mapping IDs to words.
        output_file (str): Path to the output file.
    """
    with open(output_file, 'w') as f:
        # Extract the 100 most frequent word IDs
        most_freq_ids_100 = sorted_word_ids[:100]
        for word_id in most_freq_ids_100:
            word = id2word_dict[word_id]
            f.write(f"{word}\n")
        
        # Extract 100 other IDs randomly
        random_word_ids = list(set(id2word_dict) - set(most_freq_ids_100))
        for _ in range(100):
            random_id = random.choice(random_word_ids)
            word = id2word_dict[random_id]
            f.write(f"{word}\n")

# Usage for non-filtered vocabulary
select_word_types_for_inspection(most_freq_ids_5000, id2word_dict, selected_words_path)

# Usage for controlled vocabulary
select_word_types_for_inspection(most_freq_monomorph_ids_5000, id2word_dict, controlled_selected_words_path)
