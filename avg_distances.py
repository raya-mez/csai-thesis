import os
import csv
import numpy as np
from config import Globals
import distance_funcs
import preprocess_funcs

"""Goal: get the local and global average cosine and edit distance of each word"""

# Define output directory
output_dir = os.path.join("results", "avg_distances")
os.makedirs(output_dir, exist_ok=True)

# Initialize an instance of Globals to retrieve global variables from
globals = Globals()

# Load the vocabulary
vocab = preprocess_funcs.load_vocabulary()

# Load embeddings of the words in the vocabulary
embeddings_dict = preprocess_funcs.get_vocabulary_embeddings_dict(vocab)

# # Organize word IDs by length of the correponding words for word lengths 3-7
word_ids_by_wordlength = preprocess_funcs.word_ids_by_word_length(vocab)
word_ids = []
for length, ids in word_ids_by_wordlength.items():
    word_ids.extend(ids)

# Define rescaling options for cosine distance
rescaling_options = globals.rescaling_types_dict
word_lengths = globals.word_lengths

results = {}

for rescaling, rescaling_string in rescaling_options.items():
    for word_length in word_lengths:
        # Extract the IDs of the words with the current length
        ids_current_length = word_ids_by_wordlength[word_length]
        
        # Get the corresponding words and vectors
        words = [vocab[id] for id in ids_current_length]
        vects = np.array([embeddings_dict[id] for id in ids_current_length])
        
        # Calculate the pairwise cosine and edit distances for all word pairs of the current length
        local_cos_dist_matrix = distance_funcs.cosine_distances_matrix(vects, rescaling_string)
        local_edit_dist_matrix = distance_funcs.edit_distances_matrix(words)
        
        # Calculate the average cosine and edit distance of each word of the current word length
        local_avg_cos_dists = distance_funcs.average_distances(local_cos_dist_matrix)
        local_avg_edit_dist = distance_funcs.average_distances(local_edit_dist_matrix)
        
        for id, avg_cos_dist, avg_edit_dist in zip(ids_current_length, local_avg_cos_dists, local_avg_edit_dist):
            results[vocab[id]] = {'word_length': word_length, 
                                  'id': id,
                                  'rescaling': rescaling,
                                  'local_avg_cos_dist': avg_cos_dist, 
                                  'local_avg_edit_dist': avg_edit_dist}
    
    # Compute global average distances
    # Get all words and their embeddings
    words = [vocab[id] for id in word_ids]
    vects = np.array([embeddings_dict[id] for id in word_ids])
    global_cos_dist_matrix = distance_funcs.cosine_distances_matrix(vects, rescaling_string)
    global_edit_dist_matrix = distance_funcs.edit_distances_matrix(words, norm=True)
        
    # Calculate the average cosine and edit distance of each word of the current word length
    global_avg_cos_dists = distance_funcs.average_distances(global_cos_dist_matrix)
    global_avg_edit_dist = distance_funcs.average_distances(global_edit_dist_matrix)
    
    # Store global averages for each word in results
    for id, avg_cos_dist, avg_edit_dist in zip(word_ids, global_avg_cos_dists, global_avg_edit_dist):
        results[vocab[id]]['global_avg_cos_dist'] = avg_cos_dist
        results[vocab[id]]['global_avg_edit_dist'] = avg_edit_dist
    
    output_file = os.path.join(output_dir, f"avg_dist_{rescaling}.csv")
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['word_length', 'id', 'word', 'rescaling', 'global_avg_cos_dist', 'global_avg_norm_edit_dist', 'local_avg_cos_dist', 'local_avg_raw_edit_dist'])
        # Iterate through the results dictionary
        for word, metrics in results.items():
            word_length = metrics['word_length']
            id = metrics['id']
            rescaling = metrics['rescaling']
            local_avg_cos_dist = metrics['local_avg_cos_dist']
            local_avg_edit_dist = metrics['local_avg_edit_dist']
            
            # Get global average cosine and edit distances for the word
            global_avg_cos_dist = metrics['global_avg_cos_dist']
            global_avg_norm_edit_dist = metrics['global_avg_edit_dist']
            
            # Write the data row
            csv_writer.writerow([word_length, id, word, rescaling, global_avg_cos_dist, global_avg_norm_edit_dist,
                                local_avg_cos_dist, local_avg_edit_dist])
            
    print(f"Results saved to {output_file}")