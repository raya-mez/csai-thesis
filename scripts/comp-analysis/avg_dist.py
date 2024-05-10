import sys
import logging
from os import makedirs
from gensim import models
import numpy as np
import pickle as pkl
import csv
import distance_funcs

# TODO: remove baseline cases

# --------------- Configurations ---------------
# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s : %(levelname)s : %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])

if len(sys.argv) < 6:
    logging.error("Usage: python avg_dist.py <vocab_path> <lsa_model_path> <output_folder> <cos_dist_rescaling> <baseline>")
    sys.exit(1)

# Get file paths
vocab_path = sys.argv[1] # "data/vocab.pkl"
lsa_model_path = sys.argv[2] # "models/wiki_lsi_model.model"
baseline = sys.argv[3] # 'no' / 'random' / 'binned'
cos_dist_rescaling = sys.argv[4] # 'raw' / 'mapped' / 'angular'
output_folder = sys.argv[5] # "results/[baseline]_baseline/[cos_dist_rescaling]_cos_edit_dist"

# Create output directory if it does not exist
makedirs(output_folder, exist_ok=True)

# Set a random seed for the shuffling of the vectors
np.random.seed(4242)

# Obtain correct strings for function parameters
if cos_dist_rescaling == 'mapped':
    rescaling = 'map_zero_to_one'
elif cos_dist_rescaling == 'angular':
    rescaling = 'angular_distance'
elif cos_dist_rescaling == 'raw':
    rescaling = None
else:
    logging.error(f"Unsupported value of cos_dist_rescaling: {cos_dist_rescaling}. Supported values: 'raw', 'mapped, 'angular'")

# Set default values of flags for shuffling to False
rd_shuffle_vectors = False
bin_vectors = False

# Set shuffling flags to True based on input
if baseline == "random":
    rd_shuffle_vectors = True
elif baseline == "binned":
    bin_vectors = True
elif baseline != "no":
    logging.error(f"Unsupported value of baseline: {baseline}. Supported values: 'no', 'random', 'binned'.")

logging.info(f"Setup | Rescaling string: {rescaling}; randomly shuffle vectors: {rd_shuffle_vectors}; bin vectors: {bin_vectors}")


# --------------- Functions for shuffling ---------------

def bin_vects_by_avg_cos_sim(word_ids, vects, rescaling, num_bins):
    # Calculate pairwise cosine distances
    all_pairwise_distances = distance_funcs.cosine_distances_matrix(vects, rescaling)
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

def shuffle_bins(bins):
    # Initialize a list to store the shuffled vectors
    shuffled_vects = []
    for _, data in bins.items():
        # Extract the vectors from the id:vector inner dictionary
        vectors = list(data.values())
        # Shuffle the vectors within the bin
        np.random.shuffle(vectors)
        # Append them shuffled vectors to the common list
        shuffled_vects.extend(vectors)
        
    return shuffled_vects


# --------------- Loading data ---------------
# Load the vocabulary (dictionary with id-word pairs for the 5000 most frequent words in the corpus)
logging.info("Loading vocabulary...")
with open(vocab_path, 'rb') as f:
    vocab = pkl.load(f)

# Create a dictionary storing words by length from 3 to 7 characters
logging.info(f"Filtering words by word length...")
words_ids_by_length = {length: [id for id, word in vocab.items() if len(word)==length] for length in range(3,8)}

# Load the trained LSA model
logging.info("Loading LSA model...")
lsa_model = models.LsiModel.load(lsa_model_path)


# --------------- Computing distances ---------------
for length, word_ids in words_ids_by_length.items():
    # Fetch LSA vectors for the words in the vocabulary of the current length
    vects = np.array([lsa_model.projection.u[id] for id in word_ids])
    
    if rd_shuffle_vectors:
        # Shuffle the vectors (randomly reassign vectors to words)
        np.random.shuffle(vects)
        
    if bin_vectors:
        bins = bin_vects_by_avg_cos_sim(word_ids, vects)
        vects = shuffle_bins(bins)
    
    # Compute cosine distances
    logging.info(f"Computing cosine distances for {len(word_ids)} words of length {length}...")
    cosine_distances = distance_funcs.cosine_distances_matrix(vects, rescaling=rescaling)
    
    words = [word for id, word in vocab.items() if id in word_ids]
    edit_distances = distance_funcs.edit_distances_matrix(words)
    
    # --------------- Saving pairwise distances ---------------
    # Define the output file path for pairwise distance scores
    if rd_shuffle_vectors:
        output_file_pairwise_dist = f"{output_folder}/rd_baseline_cos_edit_dist_length_{length}.csv"   
    elif bin_vectors:
        output_file_pairwise_dist = f"{output_folder}/bin_baseline_cos_edit_dist_length_{length}.csv"   
    else:
        output_file_pairwise_dist = f"{output_folder}/cos_edit_dist_length_{length}.csv"
    logging.info(f"Saving cosine and edit distances to {output_file_pairwise_dist}...")
    
    # Write the data to the output CSV file
    with open(output_file_pairwise_dist, 'w', newline='') as f:
        csv_writer = csv.writer(f)
        # Write the header row
        csv_writer.writerow(['word_length', 'word1', 'word2', 'cos_dist', 'edit_dist'])
        
        # Iterate through all pairs of words
        for i in range(len(word_ids)):
            for j in range(i+1, len(word_ids)): # Don't duplicate word pairs
                # Get the words at the corresponding indices
                word1, word2 = words[i], words[j]
                
                # Get cosine and edit distances from the corresponding matrices
                cos_dist_word = cosine_distances[i, j]
                edit_dist_word = edit_distances[i, j]
                
                # Write the words and their distances to the CSV file
                csv_writer.writerow([length, word1, word2, cos_dist_word, edit_dist_word])
    logging.info(f"Cosine and edit distances for word length {length} saved to {output_file_pairwise_dist}.")
    
    # --------------- Computing average distances for each word ---------------
    avg_cos_dist = distance_funcs.average_distances(cosine_distances)
    avg_edit_dist = distance_funcs.average_distances(edit_distances)
    
    # Define the output file path for average distance scores
    if rd_shuffle_vectors:
        output_file_avg_dist = f"{output_folder}/rd_baseline_avg_cos_edit_dist_length_{length}.csv"   
    elif bin_vectors:
        output_file_avg_dist = f"{output_folder}/bin_baseline_avg_cos_edit_dist_length_{length}.csv"   
    else:
        output_file_avg_dist = f"{output_folder}/avg_cos_edit_dist_length_{length}.csv"
    logging.info(f"Saving cosine and edit distances to {output_file_avg_dist}...")
    
    # --------------- Saving avg distances ---------------
    # Write the data to the output CSV file
    with open(output_file_avg_dist, 'w', newline='') as f:
        csv_writer = csv.writer(f)
        # Write the header row
        csv_writer.writerow(['word_length', 'word', 'avg_cos_dist', 'avg_edit_dist'])
        
        # Iterate through all words
        for i in range(len(word_ids)):
            # Get word index
            word = words[i]
            
            # Get cosine and edit distances from the corresponding arrays
            avg_cos_dist_word = avg_cos_dist[i]
            avg_edit_dist_word = avg_edit_dist[i]
            
            # Write the words and their distances to the CSV file
            csv_writer.writerow([length, word, avg_cos_dist_word, avg_edit_dist_word])
    
    logging.info(f"Cosine and edit distances for word length {length} saved to {output_file_avg_dist}.")