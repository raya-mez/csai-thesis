import sys
import logging
import nltk
# Ensure the 'words' corpus is downloaded
nltk.download('words')
from nltk.corpus import words

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

if len(sys.argv) < 3:
    print("Usage: python inspect_words.py <input_file> <output_file>")
    sys.exit(1)
    
input_file = sys.argv[1]
output_file = sys.argv[2]

# Access the word list
word_list = words.words('en')
invalid_words = []
total_words = 0

with open(input_file, 'r') as f:
    for word in f:
        word = word.strip() # Remove newline and whitespace
        total_words += 1
        if word not in word_list:
            invalid_words.append(word)

if total_words == 0:
    print("No words to process.")
    sys.exit(2)

with open(output_file, 'w') as f:
    if total_words > 0:
        ratio = len(invalid_words) / total_words
        f.write(f"Ratio of invalid over valid English words (based on nltk.corpus.words): {len(invalid_words)}/{total_words}={ratio}\n")
    else:
        f.write("No valid data to calculate ratio.\n")
    for word in invalid_words:
        f.write(f"{word}\n")