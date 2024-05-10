import sys
import bz2
import re
from gensim import corpora

bow_corpus_path = sys.argv[1]
wordids_path = sys.argv[2]
word_freq_file_path = sys.argv[3]

# Load the BoW corpus
bow_corpus = corpora.MmCorpus(bow_corpus_path)

# Load the dictionary mapping word IDs to words
word_id_dict = {} 
with bz2.open(wordids_path, 'rt', encoding='utf-8') as f:
    for line in f:
        # Extract the items in the line: word id, word, word frequency across documents
        items = re.findall(r'\b\w+\b', line)
        if len(items) >= 3:  # Ensure there are enough elements (the first line only contain a number indicating the document size)
            word = items[1]
            id = items[0] 
            word_id_dict[word] = id

# Initialize a dictionary to hold total word type frequencies
total_word_freqs = {}

# Aggregate word type frequencies across the entire corpus
for document in bow_corpus:
    for word_id, freq in document:
        total_word_freqs[word_id] = total_word_freqs.get(word_id, 0) + freq

# Sort words by frequency in descending order
sorted_word_freqs = sorted(total_word_freqs.items(), key=lambda x: x[1], reverse=True)

# Write the sorted word frequencies to a file
with open(word_freq_file_path, 'w') as f:
    for word_id, freq in sorted_word_freqs:
        word = word_id_dict[word_id]
        f.write(f"{word},{freq}\n")
