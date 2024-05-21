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
    logging.error("Usage: python compute_bin_baseline_correlations.py <vocab_type> [num_reassignments]")
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
output_file_path = os.path.join(output_folder, configs.bin_bl_corr_scores_filename)

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

# --------------- Functions for binning & shuffling  ---------------
def bin_vects_by_avg_cos_dist(word_ids, embeddings, cos_dist_type, num_bins=4):
    """
    Organizes word embeddings into bins based on their average cosine similarity.

    Parameters:
        word_ids (list): A list of word IDs corresponding to the embeddings.
        embeddings (np.ndarray): A 2D array of word embeddings, each row representing a word vector.
        cos_dist_type: How to compute the cosine distance. Options: 'raw', 'norm', 'abs', 'ang'.
        num_bins (int, optional): The number of bins to organize the vectors into. Defaults to 4.

    Returns:
        dict: A dictionary where keys are bin indices (from 0 to num_bins-1), and values are dictionaries
        mapping word IDs to their corresponding vectors within each bin.

    Description:
        The function calculates all pairwise cosine distances between word embeddings and computes the average 
        cosine distance for each embedding. Then, it sorts the vectors and word IDs based on their average distance. 
        It divides the sorted vectors and word IDs into the specified number of bins. Each bin contains a 
        dictionary where keys are word IDs and values are their corresponding word vectors.
    """    
    # Calculate pairwise cosine distances
    all_pairwise_distances = distance_funcs.cosine_distances_matrix(embeddings, cos_dist_type)
    # Calculate average cosine distance for each vector
    avg_distances = distance_funcs.average_distances(all_pairwise_distances)
    
    # Sort vectors based on their average distance and word_ids (in ascending order -> first index corresponds to smallest distance)
    sorted_indices = np.argsort(avg_distances)
    sorted_vects = embeddings[sorted_indices]
    sorted_word_ids = [word_ids[i] for i in sorted_indices]
    # print(f"Sorted word IDs: {sorted_word_ids}")
    
    # Calculate bin size such that vectors are equally distributed across all bins
    n = len(sorted_vects)
    bin_size = n // num_bins
    
    # Initialize a dictionary to represent bins. The values will be dictionaries mapping sorted word IDs to vectors
    bins = {i: {} for i in range(num_bins)}
    
    # Distribute the sorted word IDs and vectors into the bins
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
        # print(f"IDs for bin {i}: {bin_word_ids}")
        
        # Append the word IDs and corresponding vectors to the current bin
        for id, vect in zip(bin_word_ids, bin_vects):
            bins[i][id] = vect
    
    # Return the bins
    return bins

def shuffle_bins(bins):
    # Initialize a list to store the shuffled vectors
    shuffled_vects_list = []
    
    for _, data in bins.items():
        # Extract the vectors from the id:vector inner dictionary
        vectors = list(data.values())
        # Make a copy of the vectors list to ensure safe shuffling
        shuffled_vectors = vectors.copy()
        # Shuffle the copy of the vectors within the bin
        np.random.shuffle(shuffled_vectors)
        # Append the shuffled vectors to the common list
        shuffled_vects_list.extend(shuffled_vectors)
        
    # Convert the list of vectors to a numpy array
    shuffled_vects_array = np.array(shuffled_vects_list)
        
    return shuffled_vects_array


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
            
            # Bin the vectors in 4 bins based on their average cosine distance 
            bins = bin_vects_by_avg_cos_dist(ids, vects, cos_dist_type=cos_dist_type) # `bin` is a dict of the format { bin_nb: {id: vect} }
            
            # Perform random permutations of word-vector mappings within bins
            for iteration in range(num_reassignments):
                if iteration % 100 == 0:
                    logging.info(f"Reassignment iteration {iteration + 1}/{num_reassignments} for word length {wl}")
                
                # Shuffle the vectors within the bins
                shuffled_vectors = shuffle_bins(bins)
                
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