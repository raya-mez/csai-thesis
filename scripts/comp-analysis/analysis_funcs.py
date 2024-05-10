import pandas as pd
from scipy.stats import norm

def load_data(file_path):
    """
    Load the data file and return a DataFrame.
    """
    return pd.read_csv(file_path)

def compute_mean_std(df, word_length):
    """
    Compute the mean and standard deviation of transformed correlations for a specified word length.
    """
    df_filtered = df[df['word_length'] == word_length]
    mean_corr = df_filtered['transformed_corr'].mean()
    std_corr = df_filtered['transformed_corr'].std()
    return mean_corr, std_corr

def compute_z_score_p_value(real_corr, mean_corr, std_corr):
    """
    Compute the z-score and p-value for the transformed real-lexicon correlation using mean and SD of random correlations.
    """
    z_score = (real_corr - mean_corr) / std_corr
    p_value = norm.sf(abs(z_score)) * 2  # Two-tailed test
    return z_score, p_value

