import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import os

# Define file paths (you may need to adjust the paths)
data_path = "results/new_baseline"  
output_folder = "results/plots"  

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Load pre-computed average distance scores
with open(f"{data_path}/avg_cos_dist.pkl", 'rb') as f:
    avg_cosine_distances = pkl.load(f)

with open(f"{data_path}/avg_edit_dist.pkl", 'rb') as f:
    avg_edit_distances = pkl.load(f)

# Iterate over each word length category (keys in the data files)
for length in avg_cosine_distances.keys():
    # Get the average cosine distance and average edit distance arrays
    cos_dist = avg_cosine_distances[length]
    edit_dist = avg_edit_distances[length]

    # Log-transform the average cosine distance for the scatterplot
    log_cos_dist = np.log(cos_dist + 1)  # Add 1 to avoid log(0)

    # Create the scatterplot
    plt.figure()
    plt.scatter(log_cos_dist, edit_dist)
    plt.xlabel('Log-transformed Average Cosine Distance')
    plt.ylabel('Average Edit Distance')
    plt.title(f'Length {length}: Average Cosine Distance vs. Average Edit Distance')
    
    # Save the scatterplot as a file
    scatterplot_path = os.path.join(output_folder, f'scatterplot_length_{length}.png')
    plt.savefig(scatterplot_path)
    plt.close()  # Close the plot after saving

    print(f"Scatterplot for word length {length} saved to: {scatterplot_path}")

    # Get the average cosine distance arrays within each bin
    bin_avg_cos_dist = [cos_dist[bin_indices] for bin_indices in pkl.load(f"{data_path}/bin_indices_length_{length}.pkl")]

    # Create the box-and-whiskers plot
    plt.figure()
    plt.boxplot(bin_avg_cos_dist, labels=[f"Bin {i + 1}" for i in range(4)])
    plt.xlabel('Bins')
    plt.ylabel('Average Cosine Distance')
    plt.title(f'Length {length}: Average Cosine Distance by Bin')

    # Save the box-and-whiskers plot as a file
    boxplot_path = os.path.join(output_folder, f'boxplot_length_{length}.png')
    plt.savefig(boxplot_path)
    plt.close()  # Close the plot after saving

    print(f"Box plot for word length {length} saved to: {boxplot_path}")

