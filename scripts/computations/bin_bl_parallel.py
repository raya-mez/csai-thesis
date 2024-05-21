import os
import sys
import csv
import logging
import numpy as np
from gensim.models import LsiModel
from scipy.stats import pearsonr
from concurrent.futures import ProcessPoolExecutor

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
if vocab_type == 'vocab_raw':
    vocab_path = configs.vocab_path
    words_ids_by_length_path = configs.ids_by_wordlength_path
    output_folder = configs.corr_scores_dir
elif vocab_type == 'vocab_monomorph':
    vocab_path = configs.vocab_monomorph_path 
    words_ids_by_length_path = configs.ids_by_wordlength_monomorph_path
    output_folder = configs.corr_scores_monomorph_dir
else:
    logging.error(f"Invalid vocabulary type {vocab_type}. Supported values: {configs.vocab_types}")
    sys.exit(1)

os.makedirs(output_folder, exist_ok=True)
output_file_path = os.path.join(output_folder, configs.bin_bl_corr_scores_filename)

lsa_model_path = configs.lsa_model_path
cos_dist_types = configs.cos_dist_types
form_dist_types = configs.form_dist_types

# Load data
vocab = configs.load_pkl(vocab_path)
logging.info(f"Loaded {vocab_type} with {len(vocab)} items")

words_ids_by_length = configs.load_pkl(words_ids_by_length_path)
total_ids = sum(len(ids) for ids in words_ids_by_length.values())
logging.info(f"Loaded {total_ids} IDs from {words_ids_by_length_path}.")

logging.info(f"Loading LSA model...")
lsa_model = LsiModel.load(lsa_model_path)

def bin_vects_by_avg_cos_dist(word_ids, embeddings, cos_dist_type, num_bins=4):
    all_pairwise_distances = distance_funcs.cosine_distances_matrix(embeddings, cos_dist_type)
    avg_distances = distance_funcs.average_distances(all_pairwise_distances)
    sorted_indices = np.argsort(avg_distances)
    sorted_vects = embeddings[sorted_indices]
    sorted_word_ids = [word_ids[i] for i in sorted_indices]
    n = len(sorted_vects)
    bin_size = n // num_bins
    bins = {i: {} for i in range(num_bins)}
    for i in range(num_bins):
        start_idx = i * bin_size
        end_idx = n if i == num_bins-1 else (i + 1) * bin_size
        bin_vects = sorted_vects[start_idx:end_idx]
        bin_word_ids = sorted_word_ids[start_idx:end_idx]
        for id, vect in zip(bin_word_ids, bin_vects):
            bins[i][id] = vect
    return bins

def shuffle_bins(bins):
    shuffled_vects_list = []
    for _, data in bins.items():
        vectors = list(data.values())
        shuffled_vectors = vectors.copy()
        np.random.shuffle(shuffled_vectors)
        shuffled_vects_list.extend(shuffled_vectors)
    return np.array(shuffled_vects_list)

def compute_correlations(cos_dist_type, form_dist_type, wl, ids):
    vects = np.array([lsa_model.projection.u[id] for id in ids])
    words = [vocab[id] for id in ids]
    bins = bin_vects_by_avg_cos_dist(ids, vects, cos_dist_type=cos_dist_type)
    results = []
    for iteration in range(num_reassignments):
        if iteration % 100 == 0:
            logging.info(f"Iteration {iteration + 1} for word length {wl} with distances {cos_dist_type} & {form_dist_type}")
        shuffled_vectors = shuffle_bins(bins)
        cos_distance_matrix = distance_funcs.cosine_distances_matrix(shuffled_vectors, cos_dist_type=cos_dist_type)
        form_distance_matrix = distance_funcs.edit_distances_matrix(words)
        unique_cos_distances = distance_funcs.get_unique_pairwise_scores(cos_distance_matrix)
        unique_form_distances = distance_funcs.get_unique_pairwise_scores(form_distance_matrix)
        correlation, p_value = pearsonr(unique_cos_distances, unique_form_distances)
        transformed_corr = 0.5 * (np.log1p(correlation) - np.log1p(-correlation))
        results.append({
            'cos_dist_type': cos_dist_type,
            'form_dist_type': form_dist_type,
            'word_length': wl,
            'raw_correlation': correlation,
            'transformed_correlation': transformed_corr,
            'p_value': p_value
        })
    return results

all_tasks = [(cos_dist_type, form_dist_type, wl, ids) 
             for cos_dist_type in cos_dist_types 
             for form_dist_type in form_dist_types 
             for wl, ids in words_ids_by_length.items()]

all_results = []
with ProcessPoolExecutor() as executor:
    futures = [executor.submit(compute_correlations, *task) for task in all_tasks]
    for future in futures:
        all_results.extend(future.result())

with open(output_file_path, 'w', newline='') as csv_file:
    fieldnames = ['cos_dist_type', 'form_dist_type', 'word_length', 'raw_correlation', 'transformed_correlation', 'p_value']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    for result in all_results:
        writer.writerow(result)

logging.info(f"Results stored in {output_file_path}")
