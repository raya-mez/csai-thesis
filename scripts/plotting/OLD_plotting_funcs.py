import os
from os.path import basename, dirname
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import re


def scatterplot_multi(input_dir, output_path):
    """
    Generate a grid of scatterplots from CSV files in the specified input directory and save the resulting figure.
    
    Creates a grid of scatterplots, with each row representing a cosine distance type and each column representing a word length.
    The y axes are adjusted to the fit the range of possible values given the type of cosine distance. 
    The scatterplots illustrate the relationship between edit distance and cosine distance, and a linear regression line
    is fitted to each scatterplot. 
    
    Parameters:
    input_dir (str): The path to the input directory. The input directory should contain subdirectories corresponding 
                     to different versions of the cosine distance computation. Each subdirectory should contain 5 CSV 
                     files storing cosine distance and edit distance scores for a given word length (from 3 to 7). 
    output_path (str): The path to save the resulting scatterplot figure.
    
    Raises:
    ValueError: If a subdirectory name does not follow the expected format or if a CSV file name does not contain a word length.
    """
    
    # Get the list of subdirectories containing the CSV files with distance scores
    dirs = os.listdir(input_dir)
    
    # Create a list of lists with the CSV file names from each subdirectory
    # Each inner list contains files for a specific word length across the different cosine distance versions
    csv_files = np.array([[file for file in os.listdir(os.path.join(input_dir, dir))] for dir in dirs]).transpose()
    
    # Determine the number of rows and columns for the plot grid
    plot_cols = len(csv_files[0])  # Number of columns (word lengths)
    plot_rows = len(csv_files)  # Number of rows (cosine distance types)
    
    # Initialize subplots
    fig, axes = plt.subplots(nrows=plot_rows, ncols=plot_cols, figsize=(4 * plot_rows, 6 * plot_cols))
    # Set common x and y-axis labels for the entire figure
    fig.supylabel('Cosine Distance')
    fig.supxlabel('Edit Distance')
    
    # Populate the subplots with scatterplots and regression lines
    # Iterate over cosine distance version directories
    for i, dir in enumerate(dirs):
        # Validate directory name and extract cosine distance version
        cos_dist_version_match = re.search(r"^([a-z]+)", dir)
        if cos_dist_version_match is not None:
            cos_dist_version = cos_dist_version_match.group()
        else:
            raise ValueError(f"Directory {dir} has an invalid name. Expected [cos_dist_version]_cos_edit_dist.")
        
        # Get the list of files in the directory
        files = os.listdir(os.path.join(input_dir, dir))
        
        # Iterate over the files
        for j, file in enumerate(files):
            # Validate file name and extract word length
            wordlength_match = re.search(r"\d+", file)
            if wordlength_match is not None:
                wordlength = wordlength_match.group()
            else:
                raise ValueError(f"File {file} does not contain a word length.")
            
            # Get file path and read CSV data
            file_path = os.path.join(input_dir, dir, file)
            contents = pd.read_csv(file_path)
            cos_dist = contents['cos_dist']
            edit_dist = contents['edit_dist']
            
            # Calculate Pearson correlation coefficient and p-value
            pearson_corr, p_value = pearsonr(cos_dist, edit_dist)
            
            # Set y-axis limit based on cosine distance version
            if cos_dist_version == "raw":
                axes[j, i].set_ylim(0, 2)
            else:
                axes[j, i].set_ylim(0, 1)
            
            # Plot scatterplot of cosine distance vs. edit distance
            axes[j, i].scatter(edit_dist, cos_dist, alpha=0.2)
            axes[j, i].set_title(
                f"Cosine distance: {cos_dist_version}. Word length: {wordlength}\nCorrelation: {pearson_corr:.3f}, p-value: {p_value:.3e}"
            )
            
            # Fit and plot linear regression line
            slope, intercept = np.polyfit(edit_dist, cos_dist, deg=1)
            regression_line_y = slope * edit_dist + intercept
            axes[j, i].plot(edit_dist, regression_line_y, color='red', linewidth=2, label=f"y = {slope:.3f}x + {intercept:.3f}")
            
            # Add legend for the regression line
            axes[j, i].legend()
    
    # Adjust layout and save the plot
    plt.tight_layout()
    plt.savefig(output_path)


def plot_edit_vs_cos_dist(input_csv_file, output_image_dir, rescaling=None, show_plot=True):
    # --------------- Get data ---------------
    # Read data from CSV file
    data = pd.read_csv(input_csv_file)
    
    edit_dist_list = data['edit_dist']
    cost_dist_list = data['cos_dist']
    word_length = data['word_length']
    
    # Calculate the Pearson correlation coefficient between the cosine and the edit distance scores and the corresponding p-value 
    pearson_corr, p_value = pearsonr(edit_dist_list, cost_dist_list)
    
    plt.figure()
    plt.scatter(edit_dist_list, cost_dist_list, alpha=0.7)
    plt.title(f'Correlation between Edit Distance and Cosine Distance ({rescaling})\nPearson Correlation: {pearson_corr:.3f}, p-value: {p_value:.3f}')
    plt.xlabel('Edit Distance')
    plt.ylabel('Cosine Distance')
    plt.grid(True)
    
    if show_plot==True:
        plt.show()
    
    # Save the plot to a file
    plt.savefig(f"{output_image_dir}/{rescaling}_corr_plot_wordlength_{word_length}.png")
    plt.close()


def plot_avg_distances(input_csv_file, output_image_file, transforms=[]):
    """ 
    Plot and save scatterplots of average cosine and edit distances between words and their LSA embeddings.
    
    Parameters:
        input_csv_file (str): Path to the CSV file containing average distances. The file is expected to have
            columns named 'avg_edit_dist' and 'avg_cos_dist'.
        output_image_file (str): Path to save the scatterplots. The function saves up three files based on the number of values in transforms:
            - A raw scatterplot as the file specified by this parameter.
            - A log-transformed scatterplot saved as 'log_' prefixed to the output filename.
            - A z-transformed scatterplot saved as 'z_' prefixed to the output filename.
        transforms (list): List of transformations to apply to the cosine distances in the scatterplots.
            Supported values: None (no transformation), 'log' (log transformation) and 'z' (z-score transformation).
            Defaults to an empty list, which applies all transformations.
    
    Raises:
        ValueError: If an invalid transformation is provided in the 'transforms' list.
    
    """
    # Initialize transformation flags
    no_transform = False
    log_transform = False
    z_transform = False
    
    # Set flags for the transformations
    if len(transforms) == 0:
        transforms = [None, 'log', 'z']
    if None in transforms:
        no_transform = True
    if 'log' in transforms:
        log_transform = True
    if 'z' in transforms:
        z_transform = True
        
    # --------------- Get data ---------------
    # Read data from CSV file
    data = pd.read_csv(input_csv_file)
    
    # Extract the average distances from the data
    avg_edit_distance_list = data['avg_edit_dist']
    avg_cosine_distance_list = data['avg_cos_dist']
    
    # --------------- Scatterplot with raw data ---------------
    if no_transform:
        # Create a scatterplot
        plt.figure()
        plt.scatter(avg_edit_distance_list, avg_cosine_distance_list, alpha=0.7) # alpha=0.7 for more transparent points making overlapping points better discernable
        
        # Add labels and title
        plt.xlabel('Average Edit Distance')
        plt.ylabel('Average Cosine Distance')
        plt.title('Scatterplot of Average Edit Distances vs. Average Cosine Distances')
        
        # Show grid
        plt.grid(True)
        
        # Save the scatterplot to a file
        plt.savefig(output_image_file)
    
    # --------------- Scatterplot with log-transformed cosine distance ---------------
    if log_transform:
        # Log-transform the average cosine distance, adding 1 to each element to avoid log(0)
        avg_cosine_distances_log_transformed = np.log(np.array(avg_cosine_distance_list) + 1) 

        # Create a scatterplot
        plt.figure()
        plt.scatter(avg_edit_distance_list, avg_cosine_distances_log_transformed, alpha=0.7) # alpha=0.7 for more transparent points making overlapping points better discernable
        
        # Add labels and title
        plt.xlabel('Average Edit Distance')
        plt.ylabel('Log-transformed Average Cosine Distance')
        plt.title('Scatterplot of Average Edit Distances vs. Log-transformed Cosine Distances')
        
        # Show grid
        plt.grid(True)
        
        # Save the scatterplot to a file
        plt.savefig(f"{dirname(output_image_file)}/log_{basename(output_image_file)}")
    
    # --------------- Scatterplot with z-transformed cosine distance ---------------
    if z_transform:
        # Compute mean and standard deviation of average cosine distance
        mean_cosine_distance = np.mean(avg_cosine_distance_list)
        std_cosine_distance = np.std(avg_cosine_distance_list)
        
        # Z-transform the average cosine distance
        avg_cosine_distances_z_transformed = (avg_cosine_distance_list - mean_cosine_distance) / std_cosine_distance
        
        # Create a scatterplot with z-transformed cosine distance
        plt.figure()
        plt.scatter(avg_edit_distance_list, avg_cosine_distances_z_transformed, alpha=0.7)
        
        # Add labels and title
        plt.xlabel('Average Edit Distance')
        plt.ylabel('Z-transformed Average Cosine Distance')
        plt.title('Scatterplot of Average Edit Distances vs. Z-transformed Cosine Distances')
        
        # Show grid
        plt.grid(True)
        
        # Save the scatterplot to a file
        plt.savefig(f"{dirname(output_image_file)}/z_{basename(output_image_file)}")
    
    plt.close('all')