import os
import sys
import logging
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

import config

# --------------- Configurations ---------------
# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s : %(levelname)s : %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])

if len(sys.argv) < 2:
    logging.error("Usage: python compute_correlations.py <vocab_type>")
    sys.exit(1)

vocab_type = sys.argv[1] # 'vocab_raw' / 'vocab_monomorph'

configs = config.Experiment()


# Define file paths
# Raw vocabulary
if vocab_type == 'vocab_raw':
    vocab_path = configs.vocab_path
    words_ids_by_length_path = configs.ids_by_wordlength_path
    input_folder = configs.dist_scores_dir
    output_folder = configs.corr_scores_dir   

# Monomorphemic vocabulary
elif vocab_type == 'vocab_monomorph':
    vocab_path = configs.vocab_monomorph_path 
    words_ids_by_length_path = configs.ids_by_wordlength_monomorph_path
    input_folder = configs.dist_scores_monomoprh_dir
    output_folder = configs.corr_scores_monomorph_dir

else:
    logging.error(f"Invalid vocabulary type {vocab_type}. Supported values: {configs.vocab_types}")
    sys.exit(1)

# Define input and output files
input_file_path = os.path.join(input_folder, configs.pairwise_dist_scores_filename)
os.makedirs(output_folder, exist_ok=True)
output_file_path = os.path.join(output_folder, configs.real_corr_scores_filename)

# Define cosine and form distance types
cos_dist_types = configs.cos_dist_types
form_dist_types = configs.form_dist_types

# Read the data into a DataFrame
df = pd.read_csv(input_file_path)

# Prepare to collect results
results = []

# Group the data by word length
grouped = df.groupby('word_length')

# Iterate through each group
for word_length, group in grouped:
    logging.info(f'Computing correlations for word length {word_length}')
    for cos_dist_type in cos_dist_types:
        for form_dist_type in form_dist_types:
            cos_col = f"cos_dist_{cos_dist_type}"
            form_col = f"form_dist_{form_dist_type}"
            
            if cos_col in group.columns and form_col in group.columns:
                # Compute the Pearson correlation
                pearson_r, p_value = pearsonr(group[cos_col], group[form_col])
                transformed_r = 0.5 * (np.log1p(pearson_r) - np.log1p(-pearson_r))
                
                # Append the result to the results list
                results.append({
                    'word_length': word_length,
                    'cos_dist_type': cos_dist_type,
                    'form_dist_type': form_dist_type,
                    'raw_correlation': pearson_r,
                    'transformed_correlation': transformed_r,
                    'p_value': p_value
                })

# Convert results to a DataFrame
results_df = pd.DataFrame(results)

# Write the results to a CSV file
results_df.to_csv(output_file_path, index=False)

logging.info(f'Results saved to {output_file_path}')
