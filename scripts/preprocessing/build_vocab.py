import re
import sys
import bz2
import pickle
import logging
from gensim import corpora
from collections import Counter

# Configure logging to display information to the console
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s : %(levelname)s : %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])

if len(sys.argv) < 4:
    logging.error("Usage: python build_vocab.py <bow_corpus_path> <bow_corpus_path> <output_file_path>")
    sys.exit(1)

wordids_path = sys.argv[1]
bow_corpus_path = sys.argv[2]
output_file_path = sys.argv[3]

# ---------- Get word-id mappings ----------
# Create a dictionary mapping words to ids from the wiki_wordids.txt.bz2 file
id_word_dict = {}
with bz2.open(wordids_path, 'rt') as f:
    for line in f:
        match = re.match(r'(\d+)\s+(\S+)', line)
        if match:
            word_id = int(match.group(1))
            word = match.group(2)
            id_word_dict[word_id] = word
logging.info(f"Loaded id-word dictionary with {len(id_word_dict)} entries.")
logging.info(f"First 10 id-word items: {list(id_word_dict.items())[:10]}")

# ---------- Get the 5000 most frequent wordforms ----------
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

# Get 5000 most frequent word ids
logging.info("Extracting the 5000 most frequent IDs...")
most_freq_ids = [item[0] for item in total_word_freqs.most_common(5000)]
logging.info(f"First 10 most frequent word IDs: {most_freq_ids[:10]}")

# Create a list containing the 5000 most frequent words
logging.info("Creating the vocabulary...")
vocab = {id: id_word_dict[id] for id in most_freq_ids}

with open(output_file_path, 'wb') as f:
    pickle.dump(vocab, f)

