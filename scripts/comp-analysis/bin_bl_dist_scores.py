# import os
# import sys
# import logging
# import numpy as np
# import distance_funcs
# import preprocess_funcs

# # Configure logging
# logging.basicConfig(level=logging.INFO,
#                     format='%(asctime)s : %(levelname)s : %(message)s',
#                     handlers=[logging.StreamHandler(sys.stdout)])

# if len(sys.argv) < 4:
#     print("Usage: python bin_bl_dist_scores.py <vocab_path> <lsa_model_path> <output_folder>")
#     sys.exit(1)

# vocab_path = sys.argv[1] # "data/vocab.pkl"
# lsa_model_path = sys.argv[2] # "models/wiki_lsi_model.model"
# output_folder = sys.argv[3] # "results/binned_baseline/cos_edit_dist"

# # Create output directory if it does not exist
# os.makedirs(output_folder, exist_ok=True)

# # Set a random seed for shuffling reproducibility
# np.random.seed(4242)

# # --------------- Functions for binning & shuffling  ---------------
# def bin_vects_by_avg_cos_sim(word_ids, embeddings, rescaling=None, num_bins=4):
#     """
#     Organizes word embeddings into bins based on their average cosine similarity.

#     Parameters:
#         word_ids (list): A list of word IDs corresponding to the embeddings.
#         embeddings (np.ndarray): A 2D array of word embeddings, each row representing a word vector.
#         rescaling (bool, optional): Whether to rescale the embeddings.
#                                     Options: None, 'map_zero_to_one', 'angular_distance'. Defaults to None.
#         num_bins (int, optional): The number of bins to organize the vectors into. Defaults to 4.

#     Returns:
#         dict: A dictionary where keys are bin indices (from 0 to num_bins - 1), and values are dictionaries
#         mapping word IDs to their corresponding vectors within each bin.

#     Description:
#         The function calculates all pairwise cosine distances between word embeddings and computes the average 
#         cosine distance for each embedding. Then, it sorts the vectors and word IDs based on their average distance. 
#         It divides the sorted vectors and word IDs into the specified number of bins. Each bin contains a 
#         dictionary where keys are word IDs and values are their corresponding word vectors.

#     Note:
#         - The function uses `distance_funcs.pairwise_cosine_distances` and `distance_funcs.average_distances` for distance
#           calculations.
#         - In case `num_bins` is 1, the function will return one bin containing all the vectors.
#     """    
#     # Calculate pairwise cosine distances
#     all_pairwise_distances = distance_funcs.cosine_distances_matrix(embeddings, rescaling)
#     # Calculate average cosine distance for each vector
#     avg_distances = distance_funcs.average_distances(all_pairwise_distances)
    
#     # Sort vectors based on their average distance and word_ids (in ascending order -> first index corresponds to smallest distance)
#     sorted_indices = np.argsort(avg_distances)
#     sorted_vects = embeddings[sorted_indices]
#     sorted_word_ids = [word_ids[i] for i in sorted_indices]
#     # print(f"Sorted word IDs: {sorted_word_ids}")
    
#     # Calculate bin size such that vectors are equally distributed across all bins
#     n = len(sorted_vects)
#     bin_size = n // num_bins
    
#     # Initialize a dictionary to represent bins. The values will be dictionaries mapping sorted word IDs to vectors
#     bins = {i: {} for i in range(num_bins)}
    
#     # Distribute the sorted word IDs and vectors into the bins
#     for i in range(num_bins):
#         # Set the starting index for the bin
#         start_idx = i * bin_size
#         # Set the final index for the bin
#         if i == num_bins-1:  # For the last bin, include all remaining vectors
#             end_idx = n
#         else:
#             end_idx = (i + 1) * bin_size
        
#         # Slice the vectors and word IDs for the current bin
#         bin_vects = sorted_vects[start_idx:end_idx]
#         bin_word_ids = sorted_word_ids[start_idx:end_idx]
#         # print(f"IDs for bin {i}: {bin_word_ids}")
        
#         # Append the word IDs and corresponding vectors to the current bin
#         for id, vect in zip(bin_word_ids, bin_vects):
#             bins[i][id] = vect
    
#     # Return the bins
#     return bins

# def shuffle_bins(bins):
#     # Initialize a list to store the shuffled vectors
#     shuffled_vects = []
    
#     for _, data in bins.items():
#         # Extract the vectors from the id:vector inner dictionary
#         vectors = list(data.values())
#         # Shuffle the vectors within the bin
#         np.random.shuffle(vectors)
#         # Append the shuffled vectors to the common list
#         shuffled_vects.extend(vectors)
        
#     # Convert the list of vectors to a numpy array
#     shuffled_vects_array = np.array(shuffled_vects)
        
#     return shuffled_vects_array


# # --------------- Load data ---------------
# # Load the vocabulary (dictionary with id-word pairs for the 5000 most frequent words in the corpus)
# vocab = preprocess_funcs.load_vocabulary(vocab_path)

# # Get embeddings for the words in the vocabulary
# embeddings_dict = preprocess_funcs.get_vocabulary_embeddings_dict(vocab, lsa_model=lsa_model_path)

# # Create a dictionary storing words by length from 3 to 7 characters
# words_ids_by_length = {length: [id for id, word in vocab.items() if len(word)==length] for length in range(3,8)}

# # --------------- Shuffle embeddings & calculate distances (per word length) ---------------
# for wordlength, ids in words_ids_by_length.items():  
#     # Retrieve embeddings for words of the current length using their IDs
#     embeddings = np.array([embeddings_dict[id] for id in ids])
    
#     logging.info(f"Binning & shuffling for word length: {wordlength}...")  
#     # Bin the embeddings in 4 bins based on their average cosine distance 
#     bins = bin_vects_by_avg_cos_sim(ids, embeddings) # `bin` is a dict of the format { bin_nb: {id: vect} }
    
#     # Shuffle the embeddings within the bins
#     shuffled_embeddings = shuffle_bins(bins)
#     # print(shuffled_embeddings)
    
#     logging.info(f"Computing distances for word length: {wordlength}...")
#     # Calculate cosine distances of the shuffled embeddings
#     cos_distances_matrix = distance_funcs.cosine_distances_matrix(shuffled_embeddings)
    
#     # Retrieve the words correponding to the IDs for this word length
#     words = [vocab[id] for id in ids]
#     # Calculate pairwise edit distances 
#     edit_distances_matrix = distance_funcs.edit_distances_matrix(words)
    
#     # Save the cosine and edit distances to a CSV file
#     file_name = f"binned_baseline_cos_edit_dist_length_{wordlength}.csv"
#     file_path = os.path.join(output_folder, file_name)
#     logging.info(f"Saving distances for word length {wordlength} to {file_path}")
    
#     distance_funcs.save_distances(wordlength, ids, words, cos_distances_matrix, edit_distances_matrix, file_path)
    
    

###############################################################################################################
# With 10,000 bin shufflings

import os
import sys
import csv
import logging
import numpy as np
from scipy.stats import pearsonr
import distance_funcs
import preprocess_funcs

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s : %(levelname)s : %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])

if len(sys.argv) < 4:
    print("Usage: python bin_bl_dist_scores.py <vocab_path> <lsa_model_path> <output_folder>")
    sys.exit(1)

vocab_path = sys.argv[1] # "data/vocab.pkl"
lsa_model_path = sys.argv[2] # "models/wiki_lsi_model.model"
output_folder = sys.argv[3] # "results/binned_baseline"

# Create output directory if it does not exist
os.makedirs(output_folder, exist_ok=True)

# Set a random seed for reproducibility
np.random.seed(4242)

# Number of shuffling iterations
num_iterations = 10000

# --------------- Functions for binning & shuffling  ---------------
def bin_vects_by_avg_cos_sim(word_ids, embeddings, rescaling=None, num_bins=4):
    """
    Organizes word embeddings into bins based on their average cosine similarity.

    Parameters:
        word_ids (list): A list of word IDs corresponding to the embeddings.
        embeddings (np.ndarray): A 2D array of word embeddings, each row representing a word vector.
        rescaling (bool, optional): Whether to rescale the embeddings.
                                    Options: None, 'map_zero_to_one', 'angular_distance'. Defaults to None.
        num_bins (int, optional): The number of bins to organize the vectors into. Defaults to 4.

    Returns:
        dict: A dictionary where keys are bin indices (from 0 to num_bins - 1), and values are dictionaries
        mapping word IDs to their corresponding vectors within each bin.

    Description:
        The function calculates all pairwise cosine distances between word embeddings and computes the average 
        cosine distance for each embedding. Then, it sorts the vectors and word IDs based on their average distance. 
        It divides the sorted vectors and word IDs into the specified number of bins. Each bin contains a 
        dictionary where keys are word IDs and values are their corresponding word vectors.

    Note:
        - The function uses `distance_funcs.pairwise_cosine_distances` and `distance_funcs.average_distances` for distance
          calculations.
        - In case `num_bins` is 1, the function will return one bin containing all the vectors.
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
    shuffled_vects = []
    
    for _, data in bins.items():
        # Extract the vectors from the id:vector inner dictionary
        vectors = list(data.values())
        # Shuffle the vectors within the bin
        np.random.shuffle(vectors)
        # Append the shuffled vectors to the common list
        shuffled_vects.extend(vectors)
        
    # Convert the list of vectors to a numpy array
    shuffled_vects_array = np.array(shuffled_vects)
        
    return shuffled_vects_array

# Function to calculate the average correlation over multiple shuffling iterations
def calculate_avg_correlation_pvalue(bins, words, vocab):
    # Initialize a list to store the correlations for each iteration
    correlations = []
    pvalues = []
    
    # Perform 10,000 shuffling iterations
    for _ in range(num_iterations):
        # Shuffle the embeddings within the bins
        shuffled_embeddings = shuffle_bins(bins)
        
        # Calculate cosine distances of the shuffled embeddings
        cos_distances_matrix = distance_funcs.cosine_distances_matrix(shuffled_embeddings)
        
        # Calculate pairwise edit distances 
        edit_distances_matrix = distance_funcs.edit_distances_matrix(words)
        
        # Compute the correlation between the two matrices
        unique_cos_dist_array = distance_funcs.get_unique_pairwise_scores(cos_distances_matrix)
        unique_edit_dist_array = distance_funcs.get_unique_pairwise_scores(edit_distances_matrix)
        
        corr, p_value = pearsonr(unique_cos_dist_array, unique_edit_dist_array)
        
        # Store the correlation
        correlations.append(corr)
        pvalues.append(p_value)
        
    # Calculate the average correlation over all iterations
    avg_correlation = np.mean(correlations)
    avg_pvalue = np.mean(pvalues)
    
    return avg_correlation, avg_pvalue

# Load the vocabulary
vocab = preprocess_funcs.load_vocabulary(vocab_path)

# Get embeddings for the words in the vocabulary
embeddings_dict = preprocess_funcs.get_vocabulary_embeddings_dict(vocab, lsa_model=lsa_model_path)

# Create a dictionary storing words by length from 3 to 7 characters
words_ids_by_length = {length: [id for id, word in vocab.items() if len(word) == length] for length in range(3, 8)}

# Initialize a dictionary to store avg correlation for each word length
avg_corr_pvalue_per_wordlength = {}

# Iterate over each word length
for wordlength, ids in words_ids_by_length.items():  
    # Retrieve embeddings for words of the current length using their IDs
    embeddings = np.array([embeddings_dict[id] for id in ids])
    
    logging.info(f"Binning and shuffling for word length: {wordlength}...")  
    # Bin the embeddings in 4 bins based on their average cosine distance 
    bins = bin_vects_by_avg_cos_sim(ids, embeddings)
    
    # Retrieve the words corresponding to the IDs for this word length
    words = [vocab[id] for id in ids]
    
    logging.info(f"Calculating average correlation for word length: {wordlength}...")
    # Calculate the average correlation over 10,000 shuffling iterations
    avg_correlation, avg_pvalue = calculate_avg_correlation_pvalue(bins, words, vocab)
    
    # Store the average correlation for this word length in the dictionary
    avg_corr_pvalue_per_wordlength[wordlength] = (avg_correlation, avg_pvalue)
    
    # Log the average correlation
    logging.info(f"Word length {wordlength} | Avg correlation: {avg_correlation}, Avg p-value: {avg_pvalue}")

# -------------------- Save correlation scores --------------------
# Write the average correlations and p-values to the CSV file
output_file = os.path.join(output_folder, "bin_bl_corr.csv")

with open(output_file, mode="w", newline="") as f:
    writer = csv.writer(f, delimiter=",")
    
    # Write the header row
    writer.writerow(["word_length", "pearson_r", "p-value"])
    
    # Write each word length's average correlation and p-value
    for wordlength, (avg_corr, avg_p_value) in avg_corr_pvalue_per_wordlength.items():
        writer.writerow([wordlength, avg_corr, avg_p_value])

logging.info(f">>> Average correlations for random baseline saved to {output_file} <<<")
