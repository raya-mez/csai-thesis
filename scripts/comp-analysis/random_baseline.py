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
    print("Usage: python random_baseline.py <vocab_path> <lsa_model_path> <output_folder> [num_reassignments]")
    sys.exit(1)

# Read arguments
vocab_path = sys.argv[1]
lsa_model_path = sys.argv[2]
output_folder = sys.argv[3]
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

# Iterate through each rescaling option
for rescaling, rescaling_string in rescaling_options.items():
    logging.info(f"Processing for rescaling: {rescaling_string}")

    # Initialize dictionary to store results for each word length
    results = defaultdict(dict)
    
    # Iterate over word lengths
    for word_length, ids in words_ids_by_length.items():
        logging.info(f"Analyzing word length {word_length} with rescaling option: {rescaling}")
        
        # Retrieve vectors and words for the current word length
        vects = np.array([embeddings_dict[id] for id in ids])
        words = [vocab[id] for id in ids]
        
        # Initialize lists to store transformed correlations and p-values
        raw_correlations, transformed_correlations, p_values = [], [], []
        
        # Perform random permutations of word-vector mappings
        for iteration in range(num_reassignments):
            if iteration % 100 == 0:
                logging.info(f"Reassignment iteration {iteration + 1}/{num_reassignments} for word length {word_length}")
            
            # Shuffle vectors randomly
            shuffled_vectors = vects.copy()
            np.random.shuffle(shuffled_vectors)
            
            # Compute distances
            cos_distance_matrix = distance_funcs.cosine_distances_matrix(shuffled_vectors, rescaling=rescaling_string)
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
    output_file = os.path.join(output_folder, f"rd_bl_corrs_{rescaling}.csv")
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
