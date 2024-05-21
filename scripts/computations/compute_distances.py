import os
import sys
import csv
import logging
import numpy as np
from gensim.models import LsiModel

import config
import distance_funcs


# --------------- Configurations ---------------
# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s : %(levelname)s : %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])

if len(sys.argv) < 2:
    logging.error("Usage: python compute_avg_distances.py <vocab_type> <baseline>")
    sys.exit(1)

vocab_type = sys.argv[1] # 'vocab_raw' / 'vocab_monomorph'

configs = config.Experiment()


# Define file paths
# Raw vocabulary
if vocab_type == 'vocab_raw':
    vocab_path = configs.vocab_path
    words_ids_by_length_path = configs.ids_by_wordlength_path
    output_folder = configs.dist_scores_dir   

# Monomorphemic vocabulary
elif vocab_type == 'vocab_monomorph':
    vocab_path = configs.vocab_monomorph_path 
    words_ids_by_length_path = configs.ids_by_wordlength_monomorph_path
    output_folder = configs.dist_scores_monomoprh_dir

else:
    logging.error(f"Invalid vocabulary type {vocab_type}. Supported values: {configs.vocab_types}")
    sys.exit(1)

# Create output directory if it does not exist
os.makedirs(output_folder, exist_ok=True)
output_file_path = os.path.join(output_folder, configs.pairwise_dist_scores_filename)
lsa_model_path = configs.lsa_model_path

# Define cosine and form distance types
cos_dist_types = configs.cos_dist_types
form_dist_types = configs.form_dist_types


# --------------- Loading data ---------------
# Load the vocabulary (dictionary with id-word pairs for the 5000 most frequent words in the corpus)
logging.info("Loading vocabulary...")
vocab = configs.load_pkl(vocab_path)

# Load the dictionary storing lists of word IDs keyed by word length (from 3 to 7 characters)
# {3:[id1, id2,...], 4:[id3, id4,...], ...}
# words_ids_by_length = configs.group_word_ids_by_word_length(vocab)
words_ids_by_length = configs.load_pkl(words_ids_by_length_path)
total_ids = 0
for wl, ids in words_ids_by_length.items():
    total_ids += len(ids)
logging.info(f"Loaded {total_ids} IDs from {words_ids_by_length_path}.")

# Load the trained LSA model
logging.info(f"Loading LSA model...")
lsa_model = LsiModel.load(lsa_model_path)


# --------------- Computing distances ---------------
# Write the results to a file for each word pair
with open(output_file_path, "w", newline="") as csvfile:
    fieldnames = ["word_length", "word1", "word2"] + \
        [f"cos_dist_{cos_dist_type}" for cos_dist_type in cos_dist_types] + \
            [f"form_dist_{form_dist_type}" for form_dist_type in form_dist_types]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    
    # Iterate over each word combination
    for length, word_ids in words_ids_by_length.items():
        # Fetch LSA vectors and the words in the vocabulary of the current length
        logging.info(f"Processing word length {length} with {len(word_ids)} IDs.")
        vects = np.array([lsa_model.projection.u[id] for id in word_ids])
        words = [vocab[id] for id in word_ids]
        logging.info(f"Retrieved array of vectors with shape {vects.shape} and {len(words)} words.")
        
        # Generate pairwise combinations of word indices
        for i in range(len(word_ids)):
            for j in range(i+1, len(word_ids)):
                word1_id, word1 = word_ids[i], vocab[word_ids[i]]
                word2_id, word2 = word_ids[j], vocab[word_ids[j]]
                
                # Initialize row data dictionary
                row_data = {
                    "word_length": length,
                    "word1": word1,
                    "word2": word2
                }
                
                # Compute and store cosine distances for this word pair
                for cos_dist_type in cos_dist_types:
                    cos_dist = distance_funcs.cosine_distance(vects[i], vects[j], cos_dist_type)
                    row_data[f"cos_dist_{cos_dist_type}"] = cos_dist
                    
                # Compute and store form distances for this word pair
                for form_dist_type in form_dist_types:
                    form_dist = distance_funcs.form_distance(words[i], words[j], form_dist_type)
                    row_data[f"form_dist_{form_dist_type}"] = form_dist
                    
                # Write the row to the CSV file
                writer.writerow(row_data)

logging.info(f"All distances successfuly computed and saved to {output_file_path}")
