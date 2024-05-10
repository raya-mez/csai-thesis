import csv
import numpy as np
from editdistance import eval as edit_dist
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity


def compute_average_distances(words_ids_by_length, vocab, lsa_model, output_csv_file):
    """
    
    """
    # Initialize dictionaries to store sum and count for each word
    avg_edit_dist_per_word = {}
    avg_cosine_dist_per_word = {}
    
    # Iterate over each word length
    for length, word_ids in words_ids_by_length.items():
        # Fetch vectors for the words from the LSA model
        vects = np.array([lsa_model.projection.u[word_id] for word_id in word_ids])
        
        # Initialize dictionaries for the current length
        for word_id in word_ids:
            word = vocab[word_id]
            avg_edit_dist_per_word[word] = {'sum': 0, 'count': 0}
            avg_cosine_dist_per_word[word] = {'sum': 0, 'count': 0}

        # Compute cosine similarities and edit distances
        num_words = len(word_ids)
        for i in range(num_words):
            word1 = vocab[word_ids[i]]
            vector1 = vects[i]
            for j in range(i + 1, num_words):
                word2 = vocab[word_ids[j]]
                vector2 = vects[j]
                
                # Compute edit distance
                distance = edit_dist(word1, word2)
                avg_edit_dist_per_word[word1]['sum'] += distance
                avg_edit_dist_per_word[word2]['sum'] += distance
                avg_edit_dist_per_word[word1]['count'] += 1
                avg_edit_dist_per_word[word2]['count'] += 1
                
                # Compute cosine similarity and distance
                cosine_similarity = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
                minx, maxx = -1, 1
                cosine_similarity = (cosine_similarity - minx) / (maxx - minx)
                cosine_distance = 1 - cosine_similarity
                avg_cosine_dist_per_word[word1]['sum'] += cosine_distance
                avg_cosine_dist_per_word[word2]['sum'] += cosine_distance
                avg_cosine_dist_per_word[word1]['count'] += 1
                avg_cosine_dist_per_word[word2]['count'] += 1

    # Calculate average distances
    avg_edit_distances = {word: data['sum'] / data['count'] for word, data in avg_edit_dist_per_word.items()}
    avg_cosine_distances = {word: data['sum'] / data['count'] for word, data in avg_cosine_dist_per_word.items()}
    
    # Create a CSV file to store the results
    with open(output_csv_file, 'w', newline='') as csvfile:
        # Create a CSV writer object
        csv_writer = csv.writer(csvfile)
        
        # Write the header row
        csv_writer.writerow(['Word Length', 'Word', 'Average Cosine Distance', 'Average Edit Distance'])
        
        # Write the average distances for each word
        for word in avg_edit_distances:
            # Get the average edit distance and cosine distance
            avg_edit_distance = avg_edit_distances[word]
            avg_cosine_distance = avg_cosine_distances.get(word, None)
            
            # Write the data row to the CSV file
            csv_writer.writerow([length, word, avg_cosine_distance, avg_edit_distance])

    print(f"Average distances saved in {output_csv_file}")

