import logging
import sys
from gensim import corpora, models

# Configure logging to display information to the console
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s : %(levelname)s : %(message)s',
                    handlers=[logging.FileHandler("debug.log"),
                                logging.StreamHandler(sys.stdout)])

if len(sys.argv) < 5:
    print("Usage: python cos_sim.py <wordids_path> <lsa_model_path> <bow_corpus_path> <output_file_path>")
    sys.exit(1)

# wordids_path = "data/wiki_wordids.txt.bz2"
wordids_path = sys.argv[1]
tfidf_corpus_path = sys.argv[2]
lsa_model_path = sys.argv[3]
output_file_path = sys.argv[4]


# Configure logging to display information to the console
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Load the dictionary mapping words to ids
word_id_dict = corpora.Dictionary.load_from_text(wordids_path)

# Load the TF-IDF corpus
tfidf_corpus = corpora.MmCorpus(tfidf_corpus_path)

# Create the LSA model from the TF-IDF corpus
num_topics = 500  
lsi_model = models.LsiModel(tfidf_corpus, id2word=word_id_dict, num_topics=num_topics, random_seed=10)

# Save the LSA model
lsi_model.save(lsa_model_path)