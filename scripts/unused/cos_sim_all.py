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
    print("Usage: python cos_sim.py <vocab_path> <lsa_model_path> <output_folder>")
    sys.exit(1)

vocab_path = sys.argv[1]
lsa_model_path = sys.argv[2]
output_file_path = sys.argv[3]

# ---------- Load the vocabulary and the LSA model ----------
# Load the vocabulary (dictionary with id-word pairs for the 5000 most frequent words in the corpus)
with open(vocab_path, 'rb') as f:
    vocab = pkl.load(f)
logging.info(f"Vocabulary loaded with {len(vocab)} words.")

# Load the trained LSA model
logging.info("Loading LSA model...")
lsa_model = models.LsiModel.load(lsa_model_path)

# Fetch vectors for the 5000 most frequent words from the LSA model
logging.info("Fetching word vectors...")
most_freq_vects = np.array([lsa_model.projection.u[word_id] for word_id in vocab.keys()])

# ---------- Compute & save cosine similarities ----------
logging.info("Computing cosine similarities...")
# Cosine similarity = dot product of vectors / (norm of vector1 * norm of vector2)
# Calculate the dot product between each pair of vectors
cosine_similarities = np.dot(most_freq_vects, most_freq_vects.T)
# Computes the Euclidean norm (magnitude) of each vector along the rows axis, preserving the input dimensions
norms = np.linalg.norm(most_freq_vects, axis=1, keepdims=True)
# Divide the dot product by the product of the corresponding norms of the vectors involved in the dot product (through element-wise multiplication of the norm vectors)
cosine_similarities /= norms * norms.T

# Save the similarities to a file
logging.info("Saving cosine similarities...")
with open(output_file_path, 'w') as f:
    for i, word_id_i in enumerate(vocab.keys()):
        for j, word_id_j in enumerate(vocab.keys()):
            if j > i:  # Avoid duplicating pairs and self-comparison
                f.write(f"{vocab[word_id_i]},{vocab[word_id_j]},{cosine_similarities[i, j]}\n")
logging.info(f"Cosine similarities computation complete and saved to {output_file_path}.")
