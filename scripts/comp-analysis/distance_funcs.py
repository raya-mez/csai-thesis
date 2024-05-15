import csv
import logging
import numpy as np
from editdistance import eval as edit_dist

import config

configs = config.Globals()

# ---------- Cosine metrics ----------

# --- Computations for word pairs ---
def cosine_similarity(v1, v2):
    """
    Calculates the cosine similarity between two vectors.
    It is the dot product between the vectors divided by 
    the product of the magnitudes (L2 norms) of the vectors.
    
    Parameters:
        v1 (np.ndarray): The first vector.
        v2 (np.ndarray): The second vector.

    Returns:
        float: The cosine similarity between the two vectors.
    """
    # Calculate the dot product between the two vectors 
    dot_product = np.dot(v1, v2) 
    # Calculate the product of the L2 norms (magnitudes) of the two vectors
    norms = np.linalg.norm(v1) * np.linalg.norm(v2)
    # Calculate and return the cosine similarity
    return dot_product / norms


def cosine_distance(v1, v2, cos_dist_type):
    """Compute the cosine distance between two vectors.

    Args:
        v1 (np.ndarray): First vector.
        v2 (np.ndarray): Second vector.
        cos_dist_type (str): Type of cosine distance to compute. Supported values are 'abs', 'norm', and 'ang'.

    Raises:
        ValueError: If the specified `cos_dist_type` is not one of the supported types.

    Returns:
        numpy.ndarray: Cosine distance between the two input vectors.
    """
    cos_dist_types = configs.cos_dist_types
    # Check the rescaling argument and raise ValueError for invalid inputs
    if cos_dist_type not in cos_dist_types:
        raise ValueError(f"Invalid value for 'rescaling': {cos_dist_type}. Supported values are {cos_dist_types}.")
    
    # Compute cosine similarity 
    cos_sim = cosine_similarity(v1, v2)
    
    # Copmute cosine distance (depending the specified type)
    if cos_dist_type == 'abs':
        # Take the absolute value of the cosine similarities
        cos_sim_abs = np.abs(cos_sim)
        cos_dist = 1 - cos_sim_abs
    
    elif cos_dist_type == 'norm':
        # Normalize cosine similarities and compute cosine distances
        minx, maxx = -1, 1
        cos_sim_norm = (cos_sim - minx) / (maxx - minx)
        # Compute cosine distances from the cosine similaritiies (by taking their complement: `distance = 1 - similarity`)
        cos_dist = 1 - cos_sim_norm
        
    # Rescale the similarity scores as specified by the `rescaling` argument
    elif cos_dist_type == 'ang':
        # Calculate the angular distance - a formal distance metric with values ranging from 0 to 1
        angular_dist = np.arccos(cos_sim) / np.pi
        cos_dist = angular_dist
        
    else:
        # Default case: calculate cosine distances as the complement of the raw cosine similarity scores
        cos_dist = 1 - cos_sim
        
    return cos_dist

# --- Matrix computations ---
def cosine_similarities_matrix(word_vectors_array, round=10):
    """
    Calculates a matrix of cosine similarities between all pairs of word vectors.

    The cosine similarity between each pair of vectors is calculated as the dot product of the vectors normalized by the product of their L2 norms.

    Parameters:
        word_vectors (np.ndarray): A 2D numpy array where each row represents a word vector.

    Returns:
        np.ndarray: A 2D numpy array representing the cosine similarity matrix. The value at position (i, j)
                    represents the cosine similarity between the word vector at index i and the word vector at index j.
    
    Note: the similarity scores are rounded to the 10th decimal to avoid numerical precision limitations in the computations 
    and ensure the values fall within the expected range.
    """
    # Calculate the dot product between all pairs of word vectors
    cosine_similarities = np.dot(word_vectors_array, word_vectors_array.T)
    
    # Calculate the L2 norm of all word vectors
    norms = np.linalg.norm(word_vectors_array, axis=1, keepdims=True)
    
    # Normalize the dot products by the product of the norms
    cosine_similarities /= norms * norms.T
    
    if round is not None:
        cosine_similarities = np.round(cosine_similarities, round)
    
    return cosine_similarities


def is_symmetric(matrix):
    return np.array_equal(matrix, matrix.T)


def cosine_distances_matrix(word_vectors_array, cos_dist_type):
    """
    Calculates a matrix of cosine distances between all pairs of word vectors in the input array.

    The cosine distance is calculated as `1 - cosine_similarity`, where cosine similarity is computed
    using the function cosine_similarities. 
    Optionally, rescaling methods can be applied to adjust the similarity values, 
    such as converting cosine similarity to angular distance or simply mapping to the range [0,1].
    
    Parameters:
        word_vectors (np.ndarray): A 2D numpy array where each row represents a word vector.
        rescaling (str, optional): A string specifying the rescaling method to apply to cosine similarities.
            Supported values: 'abs_cos_sim', 'norm_cos_sim', 'angular_dist'. Defaults to None.

    Returns:
        np.ndarray: A 2D numpy array representing the cosine distances between each pair of word vectors.
    """
    cos_dist_types = configs.cos_dist_types
    # Check the rescaling argument and raise ValueError for invalid inputs
    if cos_dist_type not in cos_dist_types:
        raise ValueError(f"Invalid value for 'rescaling': {cos_dist_type}. Supported values are {cos_dist_types}.")
    
    # Compute matrix of cosine similarities
    cos_sim_matrix = cosine_similarities_matrix(word_vectors_array)

    if cos_dist_type == 'abs':
        # Take the absolute value of the cosine similarities
        cos_sim_matrix_abs = np.abs(cos_sim_matrix)
        cosine_distances = 1 - cos_sim_matrix_abs
    
    elif cos_dist_type == 'norm':
        # Normalize cosine similarities and compute cosine distances
        minx, maxx = -1, 1
        cos_sim_matrix_norm = (cos_sim_matrix - minx) / (maxx - minx)
        # Compute cosine distances from the cosine similaritiies (by taking their complement: `distance = 1 - similarity`)
        cosine_distances = 1 - cos_sim_matrix_norm
        
    # Rescale the similarity scores as specified by the `rescaling` argument
    elif cos_dist_type == 'ang':
        # Calculate the angular distance - a formal distance metric with values ranging from 0 to 1
        # (https://en.wikipedia.org/wiki/Cosine_similarity#Angular_distance_and_similarity) 
        angular_distances = np.arccos(cos_sim_matrix) / np.pi
        cosine_distances = angular_distances
        
    else:
        # Default case: calculate cosine distances as the complement of the raw cosine similarity scores
        cosine_distances = 1 - cos_sim_matrix
        
    return cosine_distances


def get_unique_pairwise_scores(scores_matrix):   
    """
    Retrieve the pairwise scores from the upper triangle of a symmetric matrix.
    
    Parameters:
        scores_matrix (np.ndarray): A symmetric square 2D numpy array representing pairwise scores.

    Returns:
        np.ndarray: A 1D numpy array containing the unique scores from the upper triangle of the matrix.

    """
    # Make sure the matrix is symmetrical
    assert is_symmetric(scores_matrix)
    
    # Get the indices of the upper triangle of the matrix, excluding the diagonal 
    # because the matrix is symmetrical and because we are not interested in the distance of a word to itself
    upper_triangle_indices = np.triu_indices_from(scores_matrix, k=1) 
    
    # Flatten the upper triangles of the matrix using the upper_triangle_indices
    distances_upper = scores_matrix[upper_triangle_indices]
    
    # Return an array of all pairwise distances
    return distances_upper


def average_distances(distances_matrix):
    """
    Calculate the average distance for each word to all other words in a distance matrix,
    excluding the distance from a word to itself (diagonal elements).

    Parameters:
        distances_matrix (np.ndarray): A 2D numpy array representing the distances between each pair of words or word vectors.
                                       This can be either a cosine distances matrix or an edit distances matrix.

    Returns:
        np.ndarray: An array where each value represents the average distance of a word to all other words in the input distances matrix.
    """
    # Copy the matrix to avoid modifying the original
    distances_matrix_copy = distances_matrix.copy()
    
    # Convert the matrix to a floating-point type if it isn't already
    if not np.issubdtype(distances_matrix_copy.dtype, np.floating):
        distances_matrix_copy = distances_matrix_copy.astype(float)
    
    # Exclude diagonal elements (set them to NaN)
    np.fill_diagonal(distances_matrix_copy, np.nan)
    
    # Calculate the average distance for each word (row average), ignoring NaN values
    avg_distances = np.nanmean(distances_matrix_copy, axis=1)
    
    return avg_distances


# ---------- Form metrics ----------
# --- Between word pairs ---
def edit_distance(s1, s2, norm=False):
    # Calculate edit distance between word i and word j
    if norm:
        return edit_dist(s1, s2) / max(len(s1), len(s2))
    else:
        return edit_dist(s1, s2) 

def get_unigrams_bigrams(s):
    """
    Returns a set of unigrams and adjacent bigrams from a string.
    """
    unigrams = set(s)
    bigrams = set(s[i:i+2] for i in range(len(s) - 1))
    # Combine unigrams and bigrams
    return unigrams.union(bigrams)

def jaccard_distance(s1, s2):
    """
    Calculates the Jaccard distance between two strings using unigrams and adjacent bigrams.
    
    Parameters:
        s1 (str): The first input string.
        s2 (str): The second input string.

    Returns:
        float: Jaccard distance between the two input strings.
    """
    # Get the combined set of unigrams and bigrams for each string
    set1 = get_unigrams_bigrams(s1)
    set2 = get_unigrams_bigrams(s2)
    
    # Calculate the intersection and union of the two sets
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    
    # Calculate Jaccard similarity
    jaccard_similarity = len(intersection) / len(union)
    
    # Calculate Jaccard distance
    jaccard_distance = 1 - jaccard_similarity
    
    return jaccard_distance

def form_distance(s1, s2, form_dist_type):
    """Compute the distance between two strings based on the specified metric.

    Args:
        s1 (str): First input string.
        s2 (str): Second input string.
        form_dist_type (str): Type of distance metric to compute. Supported values are 'edit', 'edit_norm', and 'jaccard'.

    Raises:
        ValueError: If the specified `form_dist_type` is not one of the supported types.

    Returns:
        float: Distance between the two input strings, based on the specified metric.
    """
    if form_dist_type not in configs.form_dist_types:
        raise ValueError(f"{form_dist_type} is not a valid argument. Possible values are: {configs.form_dist_types}")
    elif form_dist_type == 'edit':
        return edit_distance(s1, s2, norm=False)
    elif form_dist_type == 'edit_norm':
        return edit_distance(s1, s2, norm=True)
    elif form_dist_type == 'jaccard':
        return jaccard_distance(s1, s2)

# --- Using matrices ---
def edit_distances_matrix(word_list, norm=False, symmetric=True):
    """
    Compute a matrix of edit distances between each pair of words.

    Parameters:
        words (list): A list of words of equal length.
        symmetric (bool, optional): If True, assume the edit distance is symmetric. 
                                    This is the case when words are of equal length and/or when all edit operations have equal cost. 
                                    Then the function will calculate edit distance for each pair (i, j) only when i < j 
                                    and fill both distances_matrix[i, j] and distances_matrix[j, i] with the calculated distance.
                                    If False, the function computes the edit distance for each pair of words independently.
                                    Defaults to True.

    Returns:
        np.ndarray: A 2D numpy array representing the edit distances between each pair of words.
    """
    # Get the number of words
    num_words = len(word_list)
    
    # Initialize the edit distances matrix
    distances_matrix = np.zeros((num_words, num_words))
    
    if symmetric:
        # Compute edit distances for each pair of words where i < j
        # This approach leverages symmetry of edit distance to reduce computations
        for i in range(num_words):
            for j in range(i+1, num_words):
                # Calculate edit distance between word i and word j
                if norm:
                    distance = edit_dist(word_list[i], word_list[j]) / max(len(word_list[i]), len(word_list[j]))
                else:
                    distance = edit_dist(word_list[i], word_list[j])
                # Fill both distances_matrix[i, j] and distances_matrix[j, i] with the calculated distance 
                distances_matrix[i, j] = distance
                distances_matrix[j, i] = distance
    
    else:
        # Compute edit distances for each pair of words
        for i in range(num_words):
            for j in range(num_words):
                # Calculate edit distance between word at index i and word at index j
                # and fill distances_matrix[i, j] (only) with the value
                distances_matrix[i, j] = edit_dist(word_list[i], word_list[j])
    
    return distances_matrix

def jaccard_distances_matrix(word_list):
    num_words = len(word_list)
    
    # Initialize the edit distances matrix
    distances_matrix = np.zeros((num_words, num_words))
    
    # Compute jaccard distances for each pair of words where i < j
    for i in range(num_words):
        for j in range(i+1, num_words):
            # Calculate jaccard distance between word i and word j
            distance = jaccard_distance(word_list[i], word_list[j])
            # Fill both distances_matrix[i, j] and distances_matrix[j, i] with the calculated distance 
            distances_matrix[i, j] = distance
            distances_matrix[j, i] = distance
    
    return distances_matrix

def form_distances_matrix(word_list, form_dist_type):
    if form_dist_type not in configs.form_dist_types:
        raise ValueError(f"{form_dist_type} is not a valid argument. Possible values are: {configs.form_dist_types}")
    elif form_dist_type == 'edit':
        return edit_distances_matrix(word_list, norm=False)
    elif form_dist_type == 'edit_norm':
        return edit_distances_matrix(word_list, norm=True)
    elif form_dist_type == 'jaccard':
        return jaccard_distances_matrix(word_list)


# ---------- Others ----------
def check_value_range(matrix, min_value, max_value):
    """
    Check that all values in a matrix fall within a specified range.

    Parameters:
        matrix (np.ndarray): A 2D numpy array representing similarity, distance or other metric values.
        min_value (float): The minimum value of the range.
        max_value (float): The maximum value of the range.

    Returns:
        bool: True if all values in the matrix fall within the specified range, False otherwise.
    """
    # Find the minimum and maximum values in the matrix
    min_val = np.min(matrix)
    max_val = np.max(matrix)
    print(f"Min: {min_val}\nMax: {max_val}")

    # Check if both the minimum and maximum values are within the specified range
    if min_val >= min_value and max_val <= max_value:
        return True
    else:
        return False    

def save_distances(length, word_ids, words, cosine_distances_matrix, edit_distances_matrix, output_file_path):
    with open(output_file_path, 'w', newline='', encoding="utf-8") as f:
        csv_writer = csv.writer(f)
        # Write the header row
        csv_writer.writerow(['word_length', 'word1', 'word2', 'cos_dist', 'edit_dist'])
        
        # Iterate through all word pairs (all combinations of 2 words)
        for i in range(len(word_ids)):
            for j in range(i+1, len(word_ids)): # Don't duplicate word pairs
                # Get the words at the corresponding indices
                word1, word2 = words[i], words[j]
                
                # Get cosine and edit distances from the corresponding matrices
                cos_dist = cosine_distances_matrix[i, j]
                edit_dist = edit_distances_matrix[i, j]
                
                # Write the words and their distances to the CSV file
                csv_writer.writerow([length, word1, word2, cos_dist, edit_dist])
                
    logging.info(f"Cosine and edit distances for word length {length} saved to {output_file_path}.")
