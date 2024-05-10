#!/bin/bash
#SBATCH -p GPU # partition (queue)
#SBATCH -N 1 # number of nodes
#SBATCH -t 0-36:00 # time (D-HH:MM)
#SBATCH -o slurm.%N.%j.out # STDOUT
#SBATCH -e slurm.%N.%j.err # STDERR
#SBATCH --gres=gpu:1

source activate thesis

# Assumes a wiki corpus is downloaded from: wget https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2

# 1. Preprocess:
#     --> produces files (in data/): 
#         - wiki_wordids.txt.bz2: mapping between words and their integer ids
#         - wiki_bow.mm: bag-of-words (word counts) representation in Matrix Market format
#         - wiki_bow.mm.index: index for wiki_bow.mm
#         - wiki_bow.mm.metadata.cpickle: titles of documents
#         - wiki_tfidf.mm: TF-IDF representation in Matrix Market format
#         - wiki_tfidf.mm.index: index for wiki_tfidf.mm
#         - wiki.tfidf_model: TF-IDF model
echo "Started make_wikicorpus"
python -m gensim.scripts.make_wikicorpus data/enwiki-latest-pages-articles.xml.bz2 data/wiki 10000
echo "Done make_wikicorpus"

# 2. Inspect words: 
# 2.1. Extract words to inspect: get_200_word_types.sh
#     --> produces file: data/selected_wordtypes.txt
WORDIDS="data/wiki_wordids.txt.bz2"
BOW_CORPUS="data/wiki_bow.mm"
OUTPUT_FILE_PATH="data/selected_wordtypes.txt"

echo "Started extracting 200 word types"
python scripts/get_200_word_types.py $WORDIDS $BOW_CORPUS $OUTPUT_FILE_PATH
echo "Finished extracting 200 word types. Find them in data/selected_wordtypes.txt"

# 2.2. Inspect if the words are legit: inspect_words.sh
#     -> takes file: data/selected_wordtypes.txt
#     --> produces file: results/ratio_invalid_en_word_wiki.txt
INPUT_FILE_PATH="data/selected_wordtypes.txt"
OUTPUT_FILE_PATH="results/ratio_invalid_en_word_wiki.txt"

echo "Started inspecting words in data/selected_wordtypes.txt"
python scripts/inspect_words.py $INPUT_FILE_PATH $OUTPUT_FILE_PATH
echo "Finished inspecting words in data/selected_wordtypes.txt. Find the ratio of legit words in results/ratio_invalid_en_word_wiki.txt"

# 3. Run LSA model: lsa.sh
#     -> takes files (in data/): wiki_wordids.txt.bz2, wiki_tfidf.mm
#     --> produces files (in models/): wiki_lsi_model.model, wiki_lsi_model.model.projection.u.npy, wiki_lsi_model.model.projection
WORDIDS="data/wiki_wordids.txt.bz2"
TFIDF_MODEL='data/wiki_tfidf.mm'
LSA_MODEL='models/wiki_lsi_model.model'

echo "Started training LSA model"
python scripts/lsa.py $WORDIDS $TFIDF_MODEL $LSA_MODEL
echo "Finished trainign LSA model"
