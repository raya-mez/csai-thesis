import os
import sys
import csv
import logging
import numpy as np

import config
import distance_funcs


# --------------- Configurations ---------------
# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s : %(levelname)s : %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])

configs = config.Experiment()

# Get file paths
vocab_path = configs.vocab_path
lsa_model_path = configs.lsa_model_path
# output_folder = configs.dist_scores_dir
output_file_path = configs.dist_scores_file

# Create output directory if it does not exist
# os.makedirs(output_folder, exist_ok=True)

# Define rescaling options for cosine distance
cos_dist_types = configs.cos_dist_types
form_dist_types = configs.form_dist_types

# --------------- Loading data ---------------
# Load the vocabulary (dictionary with id-word pairs for the 5000 most frequent words in the corpus)
logging.info("Loading vocabulary...")
vocab = configs.load_vocabulary(vocab_path)

# Create a dictionary storing words by length from 3 to 7 characters
logging.info(f"Grouping word IDs by word length...")
words_ids_by_length = configs.word_ids_by_word_length(vocab)

# Load the trained LSA model
logging.info("Loading LSA embeddings of words in the vocabulary...")
embeddings_dict = configs.get_vocabulary_embeddings_dict(vocab, lsa_model_path)


# --------------- Computing distances ---------------
# Write the results to a file for each word pair
with open(output_file_path, "w", newline="") as csvfile:
    fieldnames = ["word_length", "word1", "word2"] + [f"cos_dict_{cos_dist_type}" for cos_dist_type in cos_dist_types] + [f"form_dist_{form_dist_type}" for form_dist_type in form_dist_types]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    
    # Iterate over each word combination
    for length, word_ids in words_ids_by_length.items():
        # Fetch LSA vectors for the words in the vocabulary of the current length
        logging.info(f"Processing word length {length} with {len(word_ids)} IDs.")
        vects = np.array([embeddings_dict[id] for id in word_ids])
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
                    logging.info(f"Computing cosine distances {cos_dist_type}...")
                    cosine_distances = distance_funcs.cosine_distances_matrix(vects, cos_dist_type)
                    logging.info(f"Obtained matrix of shape {cosine_distances.shape}. Writing the results...")
                    row_data[f"cos_dist_{cos_dist_type}"] = cosine_distances[i, j]
                    
                # Compute and store form distances for this word pair
                for form_dist_type in form_dist_types:
                    logging.info(f"Computing form distances {cos_dist_type}...")
                    form_distances = distance_funcs.form_distances_matrix(words, form_dist_type)
                    logging.info(f"Obtained matrix of shape {cosine_distances.shape}. Writing the results...")
                    row_data[f"form_dist_{form_dist_type}"] = form_distances[i, j]
                    
                # Write the row to the CSV file
                writer.writerow(row_data)

logging.info(f"All distances computed and saved to {output_file_path}")


# with open(output_file_path, 'a', newline='') as f:
#     fieldnames = ["Cosine Distance Type", "Length", "Word ID", "Word", "Cosine Distance", "Edit Distance"]
#     writer = csv.DictWriter(f, fieldnames=fieldnames)
#     writer.writeheader()
    
#     for cos_dist_type, form_dist_type in zip(cos_dist_types, form_dist_types):
#         logging.info(f"Started processing for {cos_dist_type} and {form_dist_type}")
#         # Define directory for results with current rescalign option
#         # rescaling_dir = os.path.join(output_folder, f"cos_edit_dist_{cos_dist_type}")
#         # os.makedirs(rescaling_dir, exist_ok=True)
        
#         # Compute (and save) distance scores for each word length
#         for length, word_ids in words_ids_by_length.items():
#             # Fetch LSA vectors for the words in the vocabulary of the current length
#             vects = np.array([embeddings_dict[id] for id in word_ids])
            
#             # Compute cosine distances
#             logging.info(f"Computing cosine distances for {len(word_ids)} words of length {length}...")
#             cosine_distances = distance_funcs.cosine_distances_matrix(vects, cos_dist_type=cos_dist_type)
            
#             words = [vocab[id] for id in word_ids]
#             form_distances = distance_funcs.form_distances_matrix(form_dist_type, words)
            
#             # edit_distances = distance_funcs.edit_distances_matrix(words)
#             # Write the data to the output CSV file
#             # filename = f"cos_edit_dist_{cos_dist_type}_wl_{length}.csv"
#             # output_file_path = os.path.join(rescaling_dir, filename)
#             # logging.info(f"Saving cosine and edit distances to {output_file_path}...")
#             # distance_funcs.save_distances(length, word_ids, words, cosine_distances, edit_distances, output_file_path)
            
#             logging.info(f"Processed and saved distances for words of length {length} with {cos_dist_type} and {form_dist_type}")
    
#     logging.info(f"All distances computed and saved to {output_file_path}")