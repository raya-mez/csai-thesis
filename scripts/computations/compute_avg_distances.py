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
    logging.error("Usage: python compute_avg_distances.py <vocab_type>")
    sys.exit(1)

configs = config.Experiment()

vocab_type = sys.argv[1] # 'vocab_raw' / 'vocab_monomorph'

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
output_file_path = os.path.join(output_folder, configs.avg_dist_scores_filename)
lsa_model_path = configs.lsa_model_path

# Define cosine and form distance types
cos_dist_types = configs.cos_dist_types
form_dist_types = configs.form_dist_types

# Set a random seed for the shuffling of the vectors (if applicable)
np.random.seed(4242)


# --------------- Functions for shuffling ---------------
def bin_vects_by_avg_cos_sim(word_ids, vects, cos_dict_type, num_bins):
    # Calculate pairwise cosine distances
    all_pairwise_distances = distance_funcs.cosine_distances_matrix(vects, cos_dict_type)
    # Calculate average cosine distance for each vector
    avg_distances = distance_funcs.average_distances(all_pairwise_distances)
    
    # Sort vectors based on their average distance and word_ids (in ascending order -> first index corresponds to smallest distances)
    sorted_indices = np.argsort(avg_distances)
    sorted_vects = vects[sorted_indices]
    sorted_word_ids = [word_ids[i] for i in sorted_indices]
    
    # Determine size of each bin (based on the input number of bins)
    n = len(sorted_vects)
    bin_size = n // num_bins
    
    # Initialize dictionary of dictionaries for each bin
    bins = {i: {} for i in range(num_bins)}
    
    # Distribute the sorted vectors and word IDs into the bins
    for i in range(num_bins):
        # Set the starting index for the bin
        start_idx = i * bin_size
        # Set the final index for the bin
        if i == num_bins-1:  # For the last bin, include all remaining vectors
            end_idx = n
        else:
            end_idx = (i + 1) * bin_size
        
        # Slice the vectors and word IDs for the current bin
        bin_vects = sorted_vects[start_idx:end_idx]
        bin_word_ids = sorted_word_ids[start_idx:end_idx]
        
        # Append the word IDs and corresponding vectors to the current bin
        for id, vect in zip(bin_word_ids, bin_vects):
            bins[i][id] = vect
    
    # Return the bins
    return bins


# --------------- Loading data ---------------
# Load the vocabulary (dictionary with id-word pairs for the 5000 most frequent words in the corpus)
vocab = configs.load_pkl(vocab_path)
logging.info(f"Loaded vocabulary with {len(vocab)} items")

# Load the dictionary storing lists of word IDs keyed by word length (from 3 to 7 characters)
# {3:[id1, id2,...], 4:[id3, id4,...], ...}
# words_ids_by_length = configs.group_word_ids_by_word_length(vocab)
words_ids_by_wordlength = configs.load_pkl(words_ids_by_length_path)
total_ids = 0
for wl, ids in words_ids_by_wordlength.items():
    total_ids += len(ids)
logging.info(f"Loaded {total_ids} IDs from {words_ids_by_length_path}.")

# Combine all word IDs of words of the appropriate lengths (for average computations across word lengths)
all_word_ids = []
for length, ids in words_ids_by_wordlength.items():
    all_word_ids.extend(ids)

# Load the trained LSA model
logging.info(f"Loading LSA model...")
lsa_model = LsiModel.load(lsa_model_path)


# --------------- Computing average distances ---------------
results = {}

for cos_dist_type in cos_dist_types:
    for form_dist_type in form_dist_types:
        # Compute average distances of each word within its word length
        for wl in configs.word_lengths:
            # Extract the IDs of the words with the current length
            ids_current_length = words_ids_by_wordlength[wl]
            
            # Get the corresponding words and vectors
            words = [vocab[id] for id in ids_current_length]
            vects = np.array([lsa_model.projection.u[id] for id in ids_current_length])
            
            # Calculate the pairwise cosine and form distances for all word pairs of the current length
            local_cos_dist_matrix = distance_funcs.cosine_distances_matrix(vects, cos_dist_type)
            local_form_dist_matrix = distance_funcs.form_distances_matrix(words, form_dist_type)
            
            # Calculate the average cosine and form distance of each word of the current word length
            local_avg_cos_dists = distance_funcs.average_distances(local_cos_dist_matrix)
            local_avg_form_dists = distance_funcs.average_distances(local_form_dist_matrix)
            
            # Add the results to the dictionary
            for id, avg_cos_dist, avg_form_dist in zip(ids_current_length, local_avg_cos_dists, local_avg_form_dists):
                if id not in results:
                    results[id] = {
                        'id': id,
                        'word': vocab[id],
                        'word_length': wl
                    }
                results[id][f'local_avg_cos_dist_{cos_dist_type}'] = avg_cos_dist
                results[id][f'local_avg_form_dist_{form_dist_type}'] = avg_form_dist
        
        # Compute average distances of each word across all word lengths
        all_words = [vocab[id] for id in all_word_ids]
        all_vects = np.array([lsa_model.projection.u[id] for id in all_word_ids])
        
        global_cos_dist_matrix = distance_funcs.cosine_distances_matrix(all_vects, cos_dist_type)
        global_form_dist_matrix = distance_funcs.form_distances_matrix(all_words, form_dist_type)
        
        global_avg_cos_dists = distance_funcs.average_distances(global_cos_dist_matrix)
        global_avg_form_dists = distance_funcs.average_distances(global_form_dist_matrix)
        
        # Store global averages for each word in results
        for id, avg_cos_dist, avg_form_dist in zip(all_word_ids, global_avg_cos_dists, global_avg_form_dists):
            if id not in results:
                results[id] = {
                    'id': id,
                    'word': vocab[id],
                    'word_length': len(vocab[id])
                }
            results[id][f'global_avg_cos_dist_{cos_dist_type}'] = avg_cos_dist
            results[id][f'global_avg_form_dist_{form_dist_type}'] = avg_form_dist


with open(output_file_path, mode='w', newline='') as csv_file:
    fieldnames = ['id', 'word', 'word_length']
    for cos_dist_type in cos_dist_types:
        for form_dist_type in form_dist_types:
            fieldnames.append(f'local_avg_cos_dist_{cos_dist_type}')
            fieldnames.append(f'local_avg_form_dist_{form_dist_type}')
            fieldnames.append(f'global_avg_cos_dist_{cos_dist_type}')
            fieldnames.append(f'global_avg_form_dist_{form_dist_type}')
    
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    for _, row_data in results.items():
        writer.writerow(row_data)
logging.info(f"Results saved to {output_file_path}")