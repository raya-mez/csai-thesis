import logging
import sys
import pandas as pd
import re
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# Configure logging to display information to the console
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s : %(levelname)s : %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])

if len(sys.argv) < 3:
    print("Usage: python correlation.py <scores_file_path> <corr_score_file_path> <figure_file_path>")
    sys.exit(1)

scores_file_path = sys.argv[1]
corr_score_file_path = sys.argv[2]
figure_file_path = sys.argv[3]

# Load the CSV file into a pandas DataFrame
df = pd.read_csv(scores_file_path)

# Calculate the Pearson correlation coefficient and the p-value
pearson_corr, p_value = pearsonr(df['cos_dist'], df['edit_dist'])

# Plot the correlation
plt.scatter(df['cos_dist'], df['edit_dist'], alpha=0.5)
plt.title(f'Correlation between Cosine Distance and Edit Distance\nPearson Correlation: {pearson_corr:.3f}, p-value: {p_value:.3f}')
plt.xlabel('Cosine Distance')
plt.ylabel('Edit Distance')
plt.grid(True)

# Save the plot to a file
plt.savefig(figure_file_path)

word_length_match = re.search(r"length_(\d)", scores_file_path)

# Append the result to the CSV file
with open(corr_score_file_path, 'a') as file:
    word_length = int(word_length_match.group(1))
    file.write(f'{word_length}, {pearson_corr:.3f}, {p_value:.3f}\n')