import sys
import logging
from os import makedirs
from gensim import models
import numpy as np
import pickle as pkl
import csv
import distance_funcs

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s : %(levelname)s : %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])

if len(sys.argv) < 5:
    logging.error("Usage: python random_baseline_corr.py <vocab_path> <lsa_model_path> <output_folder> <cos_dist_rescaling>")
    sys.exit(1)

vocab_path = sys.argv[1] # "data/vocab.pkl"
lsa_model_path = sys.argv[2] # "models/wiki_lsi_model.model"
cos_dist_rescaling = sys.argv[3] # 'raw' / 'mapped' / 'angular'
output_folder = sys.argv[4] # "results/random_baseline"

makedirs(output_folder, exist_ok=True)

# Set a random seed for the shuffling of the vectors
np.random.seed(4242)

if cos_dist_rescaling == 'mapped':
    rescaling = 'map_zero_to_one'
elif cos_dist_rescaling == 'angular':
    rescaling = 'angular_distance'
elif cos_dist_rescaling == 'raw':
    rescaling = None
else:
    logging.error(f"Unsupported value of cos_dist_rescaling: {cos_dist_rescaling}")

logging.info(f"Rescaling string: {rescaling}")

# Load the vocabulary (dictionary with id-word pairs for the 5000 most frequent words in the corpus)
logging.info("Loading vocabulary...")
with open(vocab_path, 'rb') as f:
    vocab = pkl.load(f)

# Load the trained LSA model
logging.info("Loading LSA model...")
lsa_model = models.LsiModel.load(lsa_model_path)

# Create a dictionary storing words by length from 3 to 7 characters
logging.info(f"Filtering words by word length...")
words_ids_by_length = {length: [id for id, word in vocab.items() if len(word)==length] for length in range(3,8)}

for length, word_ids in words_ids_by_length.items():
    # Fetch vectors for the words in the vocabulary of the current length from the LSA model
    vects = np.array([lsa_model.projection.u[id] for id in word_ids])
    
    # Shuffle the vectors (randomly reassign vectors to words)
    np.random.shuffle(vects)
    
    # Compute cosine similarities
    logging.info(f"Computing random baseline cosine distances for {len(word_ids)} words of length {length}...")
    cosine_distances = distance_funcs.cosine_distances_matrix(vects, rescaling=rescaling)
    
    words = [word for id, word in vocab.items() if id in word_ids]
    edit_distances = distance_funcs.edit_distances_matrix(words)
    
    # Save the cosine and edit distances to a file
    output_file_path = f"{output_folder}/rd_baseline_cos_edit_dist_length_{length}.csv"
    logging.info(f"Saving cosine and edit distances to {output_file_path}...")
    
    with open(output_file_path, 'w', newline='') as f:
        csv_writer = csv.writer(f)
        
        # Write the header row
        csv_writer.writerow(['word_length', 'word1', 'word2', 'cos_dist', 'edit_dist'])
        
        # Iterate through all pairs of words
        for i in range(len(word_ids)):
            for j in range(i, len(word_ids)): # Don't duplicate word pairs
                # Get the words at the corresponding indices
                word1, word2 = words[i], words[j]
                
                # Get cosine and edit distances from the corresponding matrices
                cos_dist = cosine_distances[i, j]
                edit_dist = edit_distances[i, j]
                
                # Write the words and their distances to the CSV file
                csv_writer.writerow([length, word1, word2, cos_dist, edit_dist])
                
    logging.info(f"Cosine and edit distances for word length {length} saved to {output_file_path}.")