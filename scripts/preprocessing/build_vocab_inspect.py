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

if len(sys.argv) < 5:
    logging.error("Usage: python build_vocab_inspect.py <wordids_path> <bow_corpus_path> <vocab_path> <selected_words_path>")
    sys.exit(1)

wordids_path = sys.argv[1] # "data/models/wiki_wordids.txt.bz2"
bow_corpus_path = sys.argv[2] # "data/models/wiki_bow.mm"
vocab_path = sys.argv[3] # "data/vocab.pkl"
selected_words_path = sys.argv[4] # "data/200_words.txt"


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


# ---------- Get the 5000 most frequent wordforms in the corpus ----------
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

# Get the 5000 most frequent word IDs
logging.info("Extracting the 5000 most frequent IDs...")
most_freq_ids_5000 = [item[0] for item in total_word_freqs.most_common(5000)]
logging.info(f"First 10 most frequent word IDs: {most_freq_ids_5000[:10]}")

# Create a vocabulary containing the 5000 most frequent words keyed by their IDs
logging.info("Creating the vocabulary...")
vocab = {id: id2word_dict[id] for id in most_freq_ids_5000}

# Store the vocabulary in a pickle file for easy access
with open(vocab_path, 'wb') as f:
    pkl.dump(vocab, f)


# ---------- Select 200 word types for inspection ----------
with open(selected_words_path, 'w') as f:
    # Extract the 100 most frequent word IDs
    most_freq_ids_100 = [item[0] for item in total_word_freqs.most_common(100)]
    for id in most_freq_ids_100:
        word = id2word_dict[id]
        f.write(f"{word}\n")
    
    # Extract 100 other IDs randomly
    random_word_ids = list(set(total_word_freqs.keys()) - set(most_freq_ids_100))
    for _ in range(100):
        random_id = random.choice(random_word_ids)
        word = id2word_dict[random_id]
        f.write(f"{word}\n")