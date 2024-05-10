import csv
import logging
import numpy as np
import distance_funcs
from scipy.stats import pearsonr

# Configure logging
logging.basicConfig(level=logging.INFO)

def get_rd_baseline_corr_dict(vocab, words_ids_by_length, embeddings, n=10000):
    """
    Calculate the average Pearson correlation coefficient and p-value for each word length between cosine distances
    (raw) of randomly shuffled vectors and edit distances of corresponding words.
    
    Parameters:
        vocab (dict): A dictionary mapping word IDs to words.
        words_ids_by_length (dict): A dictionary mapping word lengths (3-7) to a list of word IDs of that length.
        lsa_model: An LSA model that maps word IDs to vectors.
        n (int, optional): The number of random reassignments to perform. Defaults to 10,000.
        
    Returns:
        dict: A dictionary mapping each word length to a tuple of average correlation coefficient and p-value.
    """
    # Initialize a dictionary to store avg correlation for each word length
    avg_corr_pvalue_per_wordlength = {}

    # Repeat for each word length
    for wordlength, ids in words_ids_by_length.items(): 
        logging.info(f"Computing baseline for word length {wordlength}") 
        # Retrieve vectors and words for the IDs of the words of the current length
        vects = np.array([embeddings[id] for id in ids])
        words = [vocab[id] for id in ids]
        
        # Initialize a list to store correlations from all random reassignments
        correlations_for_this_wordlength = []
        pvalues_for_this_wordlength = []
        
        # Perform n random reassignments
        for iteration in range(n):
            if iteration % 100 == 0:
                logging.info(f"Shuffling iteration number {iteration + 1}")
            
            # Shuffle a copy of the list of vectors
            shuffled_vectors = vects.copy()
            np.random.shuffle(shuffled_vectors)
                        
            # Compute cosine distances
            cos_distance_matrix = distance_funcs.cosine_distances_matrix(shuffled_vectors)
            # Compute edit distances
            edit_distance_matrix = distance_funcs.edit_distances_matrix(words)
            
            unique_cos_dist_array = distance_funcs.get_unique_pairwise_scores(cos_distance_matrix)
            unique_edit_dist_array = distance_funcs.get_unique_pairwise_scores(edit_distance_matrix)
            
            # Compute correlation edit and cosine distance for the given word length
            corr, p_value = pearsonr(unique_cos_dist_array, unique_edit_dist_array)
            
            # Store the correlation and p-value for the current reassignment 
            correlations_for_this_wordlength.append(corr)
            pvalues_for_this_wordlength.append(p_value)
            
            logging.debug(f"Iteration {iteration + 1}: Pearson correlation: {corr}, p-value: {p_value}")
        
        # Calculate the average correlation and p-value for this word length
        avg_corr = np.mean(correlations_for_this_wordlength)
        avg_p_value = np.mean(pvalues_for_this_wordlength)
        
        logging.info(f"Word length {wordlength}: Avg correlation: {avg_corr}, Avg p-value: {avg_p_value}")
        
        # Store the average correlation and p-value for this word length in the dictionary
        avg_corr_pvalue_per_wordlength[wordlength] = (avg_corr, avg_p_value)
        
    return avg_corr_pvalue_per_wordlength


def save_correlations(corr_dict, csv_file):
    # Write the average correlations and p-values to the CSV file
    with open(csv_file, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        
        # Write the header row
        writer.writerow(["word_length", "pearson_r", "p-value"])
        
        # Write each word length's average correlation and p-value
        for wordlength, (avg_corr, avg_p_value) in corr_dict.items():
            writer.writerow([wordlength, avg_corr, avg_p_value])
