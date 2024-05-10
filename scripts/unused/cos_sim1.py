import sys
import logging
from collections import Counter
from gensim import models, corpora
import re
import numpy as np
import bz2

# Configure logging to display information to the console
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s : %(levelname)s : %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])

if len(sys.argv) < 5:
    print("Usage: python cos_sim.py <wordids_path> <lsa_model_path> <bow_corpus_path> <output_file_path>")
    sys.exit(1)

wordids_path = sys.argv[1]
lsa_model_path = sys.argv[2]
bow_corpus_path = sys.argv[3]
output_file_path = sys.argv[4]

try:
    # ---------- Get word-id mappings ----------
    # Create a dictionary mapping words to ids from the wiki_wordids.txt.bz2 file
    id_word_dict = {}
    with bz2.open(wordids_path, 'rt') as f:
        for line in f:
            match = re.match(r'(\d+)\s+(\S+)', line)
            if match:
                word_id = int(match.group(1))
                word = match.group(2)
                id_word_dict[word_id] = word
    logging.info(f"Loaded id-word dictionary with {len(id_word_dict)} entries.")
    logging.info(f"First 10 id-word items: {list(id_word_dict.items())[:10]}")

    # ---------- Get the 5000 most frequent wordforms ----------
    # Initialize a dictionary to store word frequencies
    total_word_freqs = Counter()

    # Load the BoW corpus containing word IDs and their frequencies in each document
    bow_corpus = corpora.MmCorpus(bow_corpus_path)

    # Aggregate word type frequencies across the entire corpus
    for document in bow_corpus:
        for word_id, freq in document:
            total_word_freqs[word_id] += freq

    # Get 5000 most frequent word ids
    most_freq_ids = [item[0] for item in total_word_freqs.most_common(5000)]
    print(f"10 most frequent words IDs: {most_freq_ids[:10]}")
    
    # ---------- Get the vectors of the 5000 most frequent wordforms ----------
    # Load the trained LSA model
    lsa_model = models.LsiModel.load(lsa_model_path)

    # Extract the word vectors from the LSA model
    word_vectors = lsa_model.projection.u

    # Fetch vectors for the 5000 most frequent words from the LSA model
    most_freq_vects = np.array([word_vectors[word_id] for word_id in most_freq_ids])
    
    # ---------- Compute & save cosine similarities ----------
    # Cosine similarity = dot product of vectors / (norm of vector1 * norm of vector2)
    # Calculate the dot product between each pair of vectors
    cosine_similarities = np.dot(most_freq_vects, most_freq_vects.T)
    # Computes the Euclidean norm (magnitude) of each vector along the rows axis, preserving the input dimensions
    norms = np.linalg.norm(most_freq_vects, axis=1, keepdims=True)
    # Divide the dot product by the product of the corresponding norms of the vectors involved in the dot product (through element-wise multiplication of the norm vectors)
    cosine_similarities /= norms * norms.T

    # Save the similarities to a file
    with open(output_file_path, 'w') as f:
        for i, word_id_i in enumerate(most_freq_ids):
            for j, word_id_j in enumerate(most_freq_ids):
                if j > i:  # Avoid duplicating pairs and self-comparison
                    f.write(f"{id_word_dict[word_id_i]},{id_word_dict[word_id_j]},{cosine_similarities[i, j]}\n")
    logging.info("Cosine similarities computation complete and saved.")

except Exception as e:
    logging.exception("An error occurred during processing.")