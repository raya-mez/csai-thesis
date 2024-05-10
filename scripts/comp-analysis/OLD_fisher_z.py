import logging
import sys
import numpy as np
import csv
import math
from scipy.stats import norm
from os import getcwd

# Configure logging to display information to the console
# logging.basicConfig(level=logging.INFO,
#                     format='%(asctime)s : %(levelname)s : %(message)s',
#                     handlers=[logging.StreamHandler(sys.stdout)])

# if len(sys.argv) < 4:
#     print("Usage: python fisher_z.py <real_corr_score_file_path> <baseline_corr_score_file_path> <output_file_path>")
#     sys.exit(1)

real_corr_score_file_path = sys.argv[1] # "results/correlations/pearson_correlations.csv" #
baseline_corr_score_file_path = sys.argv[2] # "results/random_baseline/pearson_correlations.csv" # 
output_file_path = sys.argv[3] # "results/effect_size_random_baseline.csv" # 

corr_dict = {}
sample_sizes = {3: 257,
                4: 584,
                5: 744,
                6: 837,
                7: 789}

# Open the files storing the correlation scores and store the correlation values for each word length in a dictionary
with open(real_corr_score_file_path, 'r') as f_real, open(baseline_corr_score_file_path, 'r') as f_baseline:
    # Read the csv files
    reader_real = csv.reader(f_real)
    reader_baseline = csv.reader(f_baseline)
    # Skip headers
    next(reader_real)
    next(reader_baseline)
    
    # Iterate through the files to extract the correlation scores for each word length
    for real_line, baseline_line in zip(reader_real, reader_baseline):
        # Extract word length and correlation score from each line
        wordlength = int(real_line[0])
        # Ensure the word lengths match in both files
        assert wordlength == int(baseline_line[0]), "Word lengths do not match!"
        corr_score_real = float(real_line[1])
        corr_score_baseline = float(baseline_line[1])
        p_value_real = float(real_line[2])
        p_value_baseline = float(baseline_line[2])
        
        # Compute Fisher's z-transformation for the correlation coefficients
        z_real = 0.5 * math.log((1 + corr_score_real) / (1 - corr_score_real))
        z_baseline = 0.5 * math.log((1 + corr_score_baseline) / (1 - corr_score_baseline))
        
        # Calculate the difference between the Fisher's z-transformed correlation coefficients
        z_diff = z_real - z_baseline
        
        # Get sample size for the current word length (they are the same for both groups)
        n = sample_sizes[wordlength]
        SE = math.sqrt(1 / (n - 3) + 1 / (n - 3))
        
        # Calculate the z-statistic
        z_stat = z_diff / SE
        
        # Calculate the p-value using a standard normal distribution (two-tailed test)
        p_value = 2 * (1 - norm.cdf(abs(z_stat))) # cdf = cumulative distribution function (returns the probability that a value drawn from a standard normal distribution is less than or equal to the given value)
        
        # Store the correlation values, p-values, Fisher's z-transformations, and significance of the difference in the dictionary
        corr_dict[wordlength] = {
            "corr_score_real": corr_score_real,
            "p_value_real": p_value_real,
            "corr_score_baseline": corr_score_baseline,
            "p_value_baseline": p_value_baseline,
            "z_real": z_real,
            "z_baseline": z_baseline,
            "z_stat": z_stat,
            "p_value_z_stat": p_value
        }

# Write the dictionary contents to a CSV file
with open(output_file_path, 'w', newline='') as f_output:
    writer = csv.writer(f_output)

    # Write the header row
    header = ["word_length", "corr_score_real", "p_value_real", "corr_score_baseline", "p_value_baseline", "z_real", "z_baseline", "z_stat", "p_value_z_stat"]
    writer.writerow(header)
    
    # Write the data rows
    for word_length, data in corr_dict.items():
        row = [word_length] + list(data.values())
        writer.writerow(row)
