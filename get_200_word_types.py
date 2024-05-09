import sys
import logging
import bz2
import random
import re
from gensim import corpora

# Configure logging to display information to the console
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

if len(sys.argv) < 4:
    print("Usage: python process_text.py <wordids_path> <bow_corpus_path> <output_file_path>")
    sys.exit(1)

wordids_path = sys.argv[1]
bow_corpus_path = sys.argv[2]
output_file_path = sys.argv[3]

# Create a dictionary mapping words to ids from the wiki_wordids.txt.bz2 file
id_word_dict = {} 
with bz2.open(wordids_path, 'rt') as f:
    for line in f:
        match = re.match(r'(\d+)\s+(\S+)', line)
        if match:
            word_id = int(match.group(1))
            word = match.group(2)
            id_word_dict[word_id] = word

# Load the dictionary mapping word IDs to words
# id_word_dict = corpora.Dictionary.load(wordids_path, mmap=None)

print(f"Loaded id-word dictionary with {len(id_word_dict)} entries.")
print(f"First 10 id-word items: {list(id_word_dict.items())[:10]}")

# Initialize a dictionary to store word frequencies
total_word_freqs = {}

# Load the BoW corpus
bow_corpus = corpora.MmCorpus(bow_corpus_path)

# Aggregate word type frequencies across the entire corpus
for document in bow_corpus:
    for word_id, freq in document:
        try:
            total_word_freqs[word_id] += freq
        except:
            total_word_freqs[word_id] = freq

print(f"Processed {len(total_word_freqs)} words from bow_corpus.")

# Sort word ids by frequency in descending order
sorted_word_ids = [item[0] for item in sorted(total_word_freqs.items(), key=lambda x: x[1], reverse=True)]
print(f"10 most frequent words IDs: {sorted_word_ids[:10]}")

# Select words and write them to a file
with open(output_file_path, 'w') as output_file:
    # Extract the n most frequent words
    n = 100 
    taken_ids = []
    for i in range(n):
        id = sorted_word_ids[i]
        try:
            word = id_word_dict[id]
            taken_ids.append(id)
            output_file.write(f"{word}\n")
        except KeyError:
            print(f"Finding most frequent words. Word id {id} not found in id_word_dict")
            continue
    
    # Extract r other words randomly
    r = 100 
    random_word_ids = list(set(total_word_freqs.keys()) - set(taken_ids))
    for i in range(r):
        random_id = random.choice(random_word_ids)
        try:
            word = id_word_dict[random_id]
            output_file.write(f"{word}\n")
        except KeyError:
            print(f"Finding random words. Word id {random_id} not found in id_word_dict")
            continue