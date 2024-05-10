import os
import logging
import sys
from os import makedirs
import pandas as pd
import re
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import scripts.plotting.OLD_plotting_funcs

# Configure logging to display information to the console
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s : %(levelname)s : %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])

if len(sys.argv) < 4:
    logging.error("Usage: python correlations_per_length.py <dist_scores_directory> <corr_score_file> <plots_directory>")
    sys.exit(1)

dist_scores_directory = sys.argv[1]
corr_score_file = sys.argv[2]
plots_directory = sys.argv[3]
rescaling = sys.argv[4]

makedirs(plots_directory, exist_ok=True)

# Initialize a list to store correlation results
corr_results = []

# Iterate over all files with distance scores (for each word length) in the directory
for filename in os.listdir(dist_scores_directory):
    logging.info(f"Processing file {filename}")
    scores_file_path = f"{dist_scores_directory}/{filename}"
    
    # Get the word length corresponding to the current file
    word_length_match = re.search(r"\d", scores_file_path)
    word_length = int(word_length_match.group(0))
    
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(scores_file_path)
    
    # Calculate the Pearson correlation coefficient between the cosine and the edit distance scores and the corresponding p-value 
    pearson_corr, p_value = pearsonr(df['cos_dist'], df['edit_dist'])
    
    # Plot the correlation
    plt.figure()
    plt.scatter(df['cos_dist'], df['edit_dist'], alpha=0.5)
    plt.title(f'Correlation between Cosine Distance and Edit Distance\nPearson Correlation: {pearson_corr:.3f}, p-value: {p_value:.3f}')
    plt.xlabel('Cosine Distance')
    plt.ylabel('Edit Distance')
    plt.grid(True)
    
    # Save the plot to a file
    plt.savefig(f"{plots_directory}/corr_plot_wordlength_{word_length}.png")
    plt.close()
    
    # Append the results to the list
    corr_results.append({'word_length': word_length, 'pearson_corr': pearson_corr, 'p_value': p_value}) 

# Create a DataFrame from the list of correlation results
corr_results = sorted(corr_results, key=lambda x: x['word_length'])
corr_df = pd.DataFrame(corr_results)

# Write the DataFrame to the CSV file
corr_df.to_csv(corr_score_file, index=False)