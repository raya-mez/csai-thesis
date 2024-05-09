import os
import sys
import csv
import logging
import numpy as np
import pickle as pkl
import distance_funcs
from gensim import models
import preprocess_funcs


# --------------- Configurations ---------------
# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s : %(levelname)s : %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])

# Get file paths
vocab_path = "data/vocab.pkl"
lsa_model_path = "models/wiki_lsi_model.model"
output_folder = "results/distance_scores"

# Create output directory if it does not exist
os.makedirs(output_folder, exist_ok=True)

# Define rescaling options for cosine distance
rescaling_options = {
    'none': None,
    'abs': 'abs_cos_sim',
    'norm': 'norm_cos_sim',
    'ang': 'angular_dist'
}

# --------------- Loading data ---------------
# Load the vocabulary (dictionary with id-word pairs for the 5000 most frequent words in the corpus)
logging.info("Loading vocabulary...")
vocab = preprocess_funcs.load_vocabulary(vocab_path)

# Create a dictionary storing words by length from 3 to 7 characters
logging.info(f"Filtering words by word length...")
words_ids_by_length = preprocess_funcs.word_ids_by_word_length(vocab)

# Load the trained LSA model
logging.info("Loading LSA embeddings of words in the vocabulary...")
embeddings_dict = preprocess_funcs.get_vocabulary_embeddings_dict(vocab, lsa_model_path)

# --------------- Computing distances ---------------
for rescaling, rescaling_string in rescaling_options.items():
    logging.info(f"Started processing for rescaling {rescaling_string}")
    
    # Define directory for results with current rescalign option
    rescaling_dir = os.path.join(output_folder, f"cos_edit_dist_{rescaling}")
    os.makedirs(rescaling_dir, exist_ok=True)

    # Compute and save distance scores for each word length
    for length, word_ids in words_ids_by_length.items():
        # Fetch LSA vectors for the words in the vocabulary of the current length
        vects = np.array([embeddings_dict[id] for id in word_ids])
        
        # Compute cosine distances
        logging.info(f"Computing cosine distances for {len(word_ids)} words of length {length}...")
        cosine_distances = distance_funcs.cosine_distances_matrix(vects, rescaling=rescaling_string)
        
        words = [vocab[id] for id in word_ids]
        edit_distances = distance_funcs.edit_distances_matrix(words)
        
        # Write the data to the output CSV file
        filename = f"cos_edit_dist_{rescaling}_wl_{length}.csv"
        output_file_path = os.path.join(rescaling_dir, filename)
        logging.info(f"Saving cosine and edit distances to {output_file_path}...")
        distance_funcs.save_distances(length, word_ids, words, cosine_distances, edit_distances, output_file_path)