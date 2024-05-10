import os
import csv
import logging
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.stats import pearsonr


input_files_dir = "results/distance_scores"
output_folder = "results/correlations/real_corr_scores"

# Create output directory if it does not exist
os.makedirs(output_folder, exist_ok=True)


rescaling_options = {
    'none': None,
    'abs': 'abs_cos_sim',
    'norm': 'norm_cos_sim',
    'ang': 'angular_dist'
}

# Iterate through each rescaling option
for rescaling, rescaling_string in rescaling_options.items():
    # Initialize a dict to score correlations scores for each word length
    results = defaultdict(dict)
    # For each word length
    for word_length in range(3,8):
        # Load data from csv file with distance scores into a dataframe
        csvfile = os.path.join(input_files_dir, f"cos_edit_dist_{rescaling}", f"cos_edit_dist{rescaling}_wl_{word_length}.csv")
        df = pd.read_csv(csvfile)
        # Extract distance scores
        cos_distances = df['cos_dist']
        edit_distances = df['edit_dist']
        # Calculate pearson correlation between cosine and edit distances
        correlation, p_value = pearsonr(cos_distances, edit_distances)
        
        # Transform correlation using Fisher Z-transformation
        transformed_corr = 0.5 * (np.log1p(correlation) - np.log1p(-correlation))
        
        # Store the raw and transformed correlations and p-value
        results[word_length]['raw_correlation'] = correlation
        results[word_length]['transformed_correlation'] = transformed_corr
        results[word_length]['p_value'] = p_value
    
    # Save the correlations scores and p-values to a file for this rescaling type
    output_file = os.path.join(output_folder, f"real_corrs_{rescaling}.csv")
    with open(output_file, 'w', newline='', encoding="utf-8") as f:
        csv_writer = csv.writer(f)
        # Write the header row
        csv_writer.writerow(['word_length', 'raw_corr', 'transformed_corr', 'p-value'])
        
        # Write results for each word length and each permutation
        for word_length in results:
            raw_correlation = results[word_length]['raw_correlation']
            transformed_correlation = results[word_length]['transformed_correlation']
            p_value = results[word_length]['p_value']
            csv_writer.writerow([word_length, raw_correlation, transformed_correlation, p_value])
        
        logging.info(f"Correlation scores for all word lengths with rescaling option {rescaling} saved to {output_file}")
