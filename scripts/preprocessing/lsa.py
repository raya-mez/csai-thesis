import os
import sys
import logging
from gensim import corpora, models


def setup_logging():
    """Configure logging to display information to the console"""
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s : %(levelname)s : %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout)])


def validate_paths(paths):
    """Validate if the provided paths are valid and accessible."""
    for path in paths:
        if not os.path.exists(path):
            logging.error("Path %s does not exist.", path)
            return False
        if not os.path.isfile(path):
            logging.error("Path %s is not a file.", path)
            return False
        if not os.access(path, os.R_OK):
            logging.error("No read access to %s.", path)
            return False
    return True


def main():
    # ---------- Configurations ----------
    setup_logging()

    # Check if enough command-line arguments are provided
    if len(sys.argv) < 4:
        print("Usage: python lsa.py <wordids_path> <bow_corpus_path> <lsa_model_path>")
        sys.exit(1)

    # Get command-line arguments
    wordids_path = sys.argv[1] # "models/bow/wiki_wordids.txt.bz2"
    bow_corpus_path = sys.argv[2] # "models/bow/wiki_[bow/tfidf].mm"
    lsa_model_path = sys.argv[3] # "models/lsa/wiki_lsi_model.model"

    # Validate paths
    paths_to_validate = [wordids_path, bow_corpus_path]
    if not validate_paths(paths_to_validate):
        sys.exit(1)

    # ---------- Creating LSA model ----------
    # Load the dictionary mapping words to ids
    id2word_dict = corpora.Dictionary.load_from_text(wordids_path)

    # Load the BoW corpus
    bow_corpus = corpora.MmCorpus(bow_corpus_path)

    # Create LSA model from BoW corpus
    num_topics = 500  
    lsi_model = models.LsiModel(bow_corpus, id2word=id2word_dict, num_topics=num_topics, random_seed=10)

    # Save the LSA model
    lsi_model.save(lsa_model_path)


if __name__ == "__main__":
    main()