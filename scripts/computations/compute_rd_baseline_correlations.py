import os
import sys
import csv
import logging
import numpy as np
from gensim.models import LsiModel
from scipy.stats import pearsonr

import config
import distance_funcs

# --------------- Configurations ---------------
# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s : %(levelname)s : %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])

if len(sys.argv) < 2:
    logging.error("Usage: python compute_rd_baseline_correlations.py <vocab_type> [num_reassignments]")
    sys.exit(1)

vocab_type = sys.argv[1] # 'vocab_raw' / 'vocab_monomorph'
num_reassignments = int(sys.argv[2]) if len(sys.argv) > 2 else 10000

configs = config.Experiment()

# Define file paths
# Raw vocabulary
if vocab_type == 'vocab_raw':
    vocab_path = configs.vocab_path
    words_ids_by_length_path = configs.ids_by_wordlength_path
    output_folder = configs.corr_scores_dir

# Monomorphemic vocabulary
elif vocab_type == 'vocab_monomorph':
    vocab_path = configs.vocab_monomorph_path 
    words_ids_by_length_path = configs.ids_by_wordlength_monomorph_path
    output_folder = configs.corr_scores_monomorph_dir

else:
    logging.error(f"Invalid vocabulary type {vocab_type}. Supported values: {configs.vocab_types}")
    sys.exit(1)

# Create output directory if it does not exist
os.makedirs(output_folder, exist_ok=True)
output_file_path = os.path.join(output_folder, configs.rd_bl_corr_scores_filename)

# Load LSA model
lsa_model_path = configs.lsa_model_path

# Define cosine and form distance types
cos_dist_types = configs.cos_dist_types
form_dist_types = configs.form_dist_types


# --------------- Loading data ---------------
# Load the vocabulary (dictionary with id-word pairs for the 5000 most frequent words in the corpus)
vocab = configs.load_pkl(vocab_path)
logging.info(f"Loaded {vocab_type} with {len(vocab)} items")

# Load the dictionary storing lists of word IDs keyed by word length (from 3 to 7 characters)
# {3:[id1, id2,...], 4:[id3, id4,...], ...}
words_ids_by_length = configs.load_pkl(words_ids_by_length_path)
total_ids = 0
for wl, ids in words_ids_by_length.items():
    total_ids += len(ids)
logging.info(f"Loaded {total_ids} IDs from {words_ids_by_length_path}.")

# Load the trained LSA model
logging.info(f"Loading LSA model...")
lsa_model = LsiModel.load(lsa_model_path)


# --------------- Computating baseline correlations ---------------
# Initialize a list to store results
results = []

# Iterate through cosine distance and form distance types
for cos_dist_type in cos_dist_types:
    for form_dist_type in form_dist_types:
        logging.info(f"Started computing for: '{cos_dist_type}' and '{form_dist_type}'")
        
        # Iterate over word lengths and corresponding word IDs
        for wl, ids in words_ids_by_length.items():
            # Retrieve vectors and words for the current word length
            vects = np.array([lsa_model.projection.u[id] for id in ids])
            words = [vocab[id] for id in ids]
            
            # Perform random permutations of word-vector mappings within bins
            for iteration in range(num_reassignments):
                if iteration % 100 == 0:
                    logging.info(f"Reassignment iteration {iteration + 1}/{num_reassignments} for word length {wl}")
                
                # Shuffle the vectors within the bins
                shuffled_vectors = vects.copy()
                np.random.shuffle(shuffled_vectors)
                
                # Compute distances
                cos_distance_matrix = distance_funcs.cosine_distances_matrix(shuffled_vectors, cos_dist_type=cos_dist_type)
                form_distance_matrix = distance_funcs.edit_distances_matrix(words)
                
                # Extract unique pairwise scores
                unique_cos_distances = distance_funcs.get_unique_pairwise_scores(cos_distance_matrix)
                unique_form_distances = distance_funcs.get_unique_pairwise_scores(form_distance_matrix)
                
                # Compute Pearson correlation between distances
                correlation, p_value = pearsonr(unique_cos_distances, unique_form_distances)
                
                # Transform correlation using Fisher Z-transformation
                transformed_corr = 0.5 * (np.log1p(correlation) - np.log1p(-correlation))
                
                # Store each correlation and p-value separately with corresponding metadata
                results.append({
                    'cos_dist_type': cos_dist_type,
                    'form_dist_type': form_dist_type,
                    'word_length': wl,
                    'raw_correlation': correlation,
                    'transformed_correlation': transformed_corr,
                    'p_value': p_value
                })

# Write results to a CSV file
with open(output_file_path, 'w', newline='') as csv_file:
    fieldnames = ['cos_dist_type', 'form_dist_type', 'word_length', 'raw_correlation', 'transformed_correlation', 'p_value']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    
    for result in results:
        writer.writerow(result)

logging.info(f"Results stored in {output_file_path}")