import sys
import logging
from gensim import models
import numpy as np
import pickle as pkl

# Configure logging to display information to the console
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s : %(levelname)s : %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])

if len(sys.argv) < 4:
    print("Usage: python cos_sim_wordlength.py <vocab_path> <lsa_model_path> <output_folder>")
    sys.exit(1)

vocab_path = sys.argv[1]
lsa_model_path = sys.argv[2]
output_folder = sys.argv[3]

# Load the vocabulary (dictionary with id-word pairs for the 5000 most frequent words in the corpus)
with open(vocab_path, 'rb') as f:
    vocab = pkl.load(f)

# Load the trained LSA model
logging.info("Loading LSA model...")
lsa_model = models.LsiModel.load(lsa_model_path)

# Create a dictionary storing words by length from 3 to 7 characters
words_ids_by_length = {length: [id for id, word in vocab.items() if len(word)==length] for length in range(3,8)}

for length, word_ids in words_ids_by_length.items():
    logging.info(f"Computing cosine similarities for {len(word_ids)} words of length {length}...")
    
    # Fetch vectors for the words in the vocabulary of the current length from the LSA model
    vects = np.array([lsa_model.projection.u[id] for id in word_ids])
    
    # Compute cosine similarities
    cosine_similarities = np.dot(vects, vects.T)
    norms = np.linalg.norm(vects, axis=1, keepdims=True)
    cosine_similarities /= norms * norms.T
    # Adjust the scores to range from 0 to 1
    minx = -1 
    maxx = 1
    cosine_similarities = (cosine_similarities-minx) / (maxx-minx)
    
    # Compute cosine distances from the cosine similaritiies
    cosine_distances = 1 - cosine_similarities
        
    # Compute edit distances and save distance scores to a file
    output_file_path = f"{output_folder}/cos_dist_length_{length}.csv"
    logging.info(f"Saving cosine distances to {output_file_path}...")
    with open(output_file_path, 'w') as f:
        for i, word_id_i in enumerate(word_ids):
            for j, word_id_j in enumerate(word_ids):
                if j > i:  # Avoid duplicating pairs and self-comparison
                    f.write(f"{vocab[word_id_i]},{vocab[word_id_j]},{cosine_distances[i, j]}\n")
    logging.info(f"Cosine distances for words of length {length} saved to {output_file_path}")

logging.info("Cosine distances computation complete and saved.")