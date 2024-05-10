"""
Performs a correlation analysis between cosine distances and edit distances for words of lengths 3 to 7.

Main steps:
- Loads vocabulary and retrieves LSA word embeddings for the words in the vocabulary.
- For each word length (from 3 to 7 characters):
    - Retrieves word vectors and word list for that length.
    - Shuffles word vectors and calculates cosine distances and edit distances between all word pairs.
    - Computes and stores correlations between cosine and edit distances for all word pairs.
    - Repeats the process for a specified number of iterations (defaults to 10,000).
    - Computes the average correlation and p-value between cosine and edit distances over all shuffling iterations for the given word length.
- Save the average correlation results for each word length to a CSV file.

External dependencies:
- `distance_funcs` for distance calculations.
- `preprocess_funcs` for data loading and pre-processing.
- `scipy.stats.pearsonr` for computing Pearson correlation coefficients.
- `numpy` for numerical operations and random shuffling.
- others: `os`, `sys`, `csv`, `logging`.

Script Configuration:
- Vocabulary file path, LSA model path, number of iterations, and output file path are defined as constants.
- Random seed is set for reproducibility.
"""

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
    print("Usage: python rd_baseline.py <vocab_path> <lsa_model_path> <output_folder> <num_iterations>")
    sys.exit(1)

vocab_path = sys.argv[1] # "data/vocab.pkl"
lsa_model_path = sys.argv[2] # "models/wiki_lsi_model.model"
output_file = sys.argv[3] # os.path.join("results", "rd_bl_corr.csv")

# Set default number of random reassignments
if len(sys.argv) > 4:
    num_iterations = sys.argv[4] # Number of random word-vectors reassignments
else:
    num_iterations = 10000

# Set a random seed for shuffling reproducibility
np.random.seed(4242)


# -------------------- Load data --------------------
# Load the vocabulary (dictionary with id-word pairs for the 5000 most frequent words in the corpus)
vocab = preprocess_funcs.load_vocabulary(vocab_path)

# Get embeddings for the words in the vocabulary
embeddings_dict = preprocess_funcs.get_vocabulary_embeddings_dict(vocab, lsa_model=lsa_model_path)

# Create a dictionary storing words by length from 3 to 7 characters
words_ids_by_length = preprocess_funcs.word_ids_by_word_length(vocab)


# -------------------- Shuffle vectors, compute distances & correlations --------------------
# Initialize a dictionary to store avg correlation for each word length
avg_corr_pvalue_per_wordlength = {}

# Repeat for each word length
for wordlength, ids in words_ids_by_length.items(): 
    logging.info(f"Computing baseline for word length {wordlength}") 
    # Retrieve vectors and words for the current IDs
    vects = np.array([embeddings_dict[id] for id in ids])
    words = [vocab[id] for id in ids]
    
    # Initialize a list to store correlations from all random reassignments for this word length
    correlations_for_this_wordlength = []
    pvalues_for_this_wordlength = []
    
    # Perform n random reassignments
    for iteration in range(num_iterations):
        if iteration % 100 == 0:
            logging.info(f"Shuffling iteration number {iteration + 1}")
        
        # Shuffle a copy of the list of vectors
        shuffled_vectors = vects.copy()
        np.random.shuffle(shuffled_vectors)
                    
        # Compute cosine distances
        # NOTE: Uses raw cosine distances
        cos_distance_matrix = distance_funcs.cosine_distances_matrix(shuffled_vectors, rescaling=None)
        # Compute edit distances
        edit_distance_matrix = distance_funcs.edit_distances_matrix(words)
        
        unique_cos_dist_array = distance_funcs.get_unique_pairwise_scores(cos_distance_matrix)
        unique_edit_dist_array = distance_funcs.get_unique_pairwise_scores(edit_distance_matrix)
        
        # Compute correlation edit and cosine distance for the given word length
        corr, p_value = pearsonr(unique_cos_dist_array, unique_edit_dist_array)
        
        # Store the correlation and p-value for the current reassignment 
        correlations_for_this_wordlength.append(corr)
        pvalues_for_this_wordlength.append(p_value)
    
    # Calculate the average correlation and p-value for this word length
    avg_corr = np.mean(correlations_for_this_wordlength)
    avg_p_value = np.mean(pvalues_for_this_wordlength)
    
    logging.info(f"Word length {wordlength} | Avg correlation: {avg_corr}, Avg p-value: {avg_p_value}")
    
    # Store the average correlation and p-value for this word length in the dictionary
    avg_corr_pvalue_per_wordlength[wordlength] = (avg_corr, avg_p_value)
    

# -------------------- Save correlation scores --------------------
# Write the average correlations and p-values to the CSV file
with open(output_file, mode="w", newline="") as f:
    writer = csv.writer(f, delimiter=",")
    
    # Write the header row
    writer.writerow(["word_length", "pearson_r", "p-value"])
    
    # Write each word length's average correlation and p-value
    for wordlength, (avg_corr, avg_p_value) in avg_corr_pvalue_per_wordlength.items():
        writer.writerow([wordlength, avg_corr, avg_p_value])

logging.info(f">>> Average correlations for random baseline saved to {output_file} <<<")