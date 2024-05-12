import sys
import logging
from gensim import models
import numpy as np
import pickle as pkl

# Configure logging to display information to the console
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s : %(levelname)s : %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])

if len(sys.argv) < 4:
    print("Usage: python sample_vectors.py <vocab_path> <lsa_model_path> <output_file>")
    sys.exit(1)

vocab_path = sys.argv[1]
lsa_model_path = sys.argv[2]
output_file = sys.argv[3]

# Load the vocabulary (dictionary with id-word pairs for the 5000 most frequent words in the corpus)
logging.info("Loading the vocabulary...")
with open(vocab_path, 'rb') as f:
    vocab = pkl.load(f)

# Load the trained LSA model
logging.info("Loading LSA model...")
lsa_model = models.LsiModel.load(lsa_model_path)

# Fetch vectors for the first 100 words in the vocabulary
first_100_words = list(vocab.items())[:100]
first_100_ids = [word_id for word_id, _ in first_100_words]
first_100_words_text = [word for _, word in first_100_words]
vects = np.array([lsa_model.projection.u[word_id] for word_id in first_100_ids])

# Save the data in npz format
logging.info(f"Saving the data in {output_file}...")
np.savez(output_file, ids=first_100_ids, words=first_100_words_text, vectors=vects)