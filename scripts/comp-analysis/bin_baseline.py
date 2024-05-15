import os
import sys
import csv
import logging
import numpy as np
from scipy.stats import pearsonr, norm
from collections import defaultdict
import distance_funcs
import preprocess_funcs

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s : %(levelname)s : %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])

if len(sys.argv) < 4:
    print("Usage: python bin_baseline.py <vocab_path> <lsa_model_path> <output_folder> [num_reassignments]")
    sys.exit(1)

# Read arguments
vocab_path = sys.argv[1] # "data/vocab.pkl"
lsa_model_path = sys.argv[2] # "models/wiki_lsi_model.model"
output_folder = sys.argv[3] # "results/correlations/bin_baseline"
num_reassignments = int(sys.argv[4]) if len(sys.argv) > 4 else 10000

# Create output directory if it does not exist
os.makedirs(output_folder, exist_ok=True)

# Set a random seed for reproducibility
np.random.seed(4242)

# Load the vocabulary and embeddings
vocab = preprocess_funcs.load_vocabulary(vocab_path)
embeddings_dict = preprocess_funcs.get_vocabulary_embeddings_dict(vocab, lsa_model=lsa_model_path)

# Create a dictionary storing words by length (3-7)
words_ids_by_length = preprocess_funcs.word_ids_by_word_length(vocab)

# Define rescaling options for cosine distance
rescaling_options = {
    'none': None,
    'abs': 'abs_cos_sim',
    'norm': 'norm_cos_sim',
    'ang': 'angular_dist'
}

# --------------- Functions for binning & shuffling  ---------------
def bin_vects_by_avg_cos_dist(word_ids, embeddings, rescaling, num_bins=4):
    """
    Organizes word embeddings into bins based on their average cosine similarity.

    Parameters:
        word_ids (list): A list of word IDs corresponding to the embeddings.
        embeddings (np.ndarray): A 2D array of word embeddings, each row representing a word vector.
        rescaling (optional): Whether to rescale the embeddings.
                                    Options: None, 'abs_cos_sim', 'norm_cos_sim', 'angular_dist'.
        num_bins (int, optional): The number of bins to organize the vectors into. Defaults to 4.

    Returns:
        dict: A dictionary where keys are bin indices (from 0 to num_bins - 1), and values are dictionaries
        mapping word IDs to their corresponding vectors within each bin.

    Description:
        The function calculates all pairwise cosine distances between word embeddings and computes the average 
        cosine distance for each embedding. Then, it sorts the vectors and word IDs based on their average distance. 
        It divides the sorted vectors and word IDs into the specified number of bins. Each bin contains a 
        dictionary where keys are word IDs and values are their corresponding word vectors.

    Note: In case `num_bins` is 1, the function will return one bin containing all the vectors.
    """    
    # Calculate pairwise cosine distances
    all_pairwise_distances = distance_funcs.cosine_distances_matrix(embeddings, rescaling)
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

# --------------- Computations ---------------
# Iterate through each rescaling option
for rescaling, rescaling_string in rescaling_options.items():
    logging.info(f"Processing for rescaling: {rescaling_string}")

    # Initialize dictionary to store results for each word length
    results = {wordlength: {
        'raw_correlations': [],
        'transformed_correlations': [],
        'p_values': []}
               for wordlength in range(3,8)}
    
    # Iterate over word lengths
    for word_length, ids in words_ids_by_length.items():
        logging.info(f"Analyzing word length {word_length} with rescaling option: {rescaling}")
        
        # Retrieve vectors and words for the current word length
        vects = np.array([embeddings_dict[id] for id in ids])
        words = [vocab[id] for id in ids]
        
        # Bin the vectors in 4 bins based on their average cosine distance 
        bins = bin_vects_by_avg_cos_dist(ids, vects, rescaling=rescaling_string) # `bin` is a dict of the format { bin_nb: {id: vect} }
        
        # Initialize lists to store raw and transformed correlations and p-values
        raw_correlations, transformed_correlations, p_values = [], [], []
        
        # Perform random permutations of word-vector mappings within bins
        for iteration in range(num_reassignments):
            if iteration % 100 == 0:
                logging.info(f"Reassignment iteration {iteration + 1}/{num_reassignments} for word length {word_length}")
            
            # Shuffle the vectors within the bins
            shuffled_vectors = shuffle_bins(bins)
            
            # Compute distances
            cos_distance_matrix = distance_funcs.cosine_distances_matrix(shuffled_vectors, cos_dist_type=rescaling_string)
            edit_distance_matrix = distance_funcs.edit_distances_matrix(words)
            
            # Extract unique pairwise scores
            unique_cos_distances = distance_funcs.get_unique_pairwise_scores(cos_distance_matrix)
            unique_edit_distances = distance_funcs.get_unique_pairwise_scores(edit_distance_matrix)
            
            # Compute Pearson correlation between distances
            correlation, p_value = pearsonr(unique_cos_distances, unique_edit_distances)
            
            # Transform correlation using Fisher Z-transformation
            transformed_corr = 0.5 * (np.log1p(correlation) - np.log1p(-correlation))
            
            # Store the raw and transformed correlation and p-value
            raw_correlations.append(correlation)
            transformed_correlations.append(transformed_corr)
            p_values.append(p_value)
        
        results[word_length]['raw_correlations'] = raw_correlations
        results[word_length]['transformed_correlations'] = transformed_correlations
        results[word_length]['p_values'] = p_values
    
    # Save results to a CSV file: word_length, transformed_corr
    output_file = os.path.join(output_folder, f"bin_bl_corrs_{rescaling}.csv")
    with open(output_file, 'w', newline='', encoding="utf-8") as f:
        csv_writer = csv.writer(f)
        # Write the header row
        csv_writer.writerow(['word_length', 'raw_corr', 'transformed_corr', 'p-value'])
        
        # Write results for each word length and each permutation
        for word_length in results:
            raw_corrs = results[word_length]['raw_correlations']
            transformed_corrs = results[word_length]['transformed_correlations']
            pvalues = results[word_length]['p_values']
            for raw_corr, transf_corr, p_value in zip(raw_corrs, transformed_corrs, pvalues):
                csv_writer.writerow([word_length, raw_corr, transf_corr, p_value])
        
        logging.info(f"Correlation scores for all word lengths with rescaling option {rescaling} saved to {output_file}")

logging.info("All rescaling options have been processed.")