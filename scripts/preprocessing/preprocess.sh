#!/bin/bash
#SBATCH -p GPU # partition (queue)
#SBATCH -N 1 # number of nodes
#SBATCH -t 0-36:00 # time (D-HH:MM)
#SBATCH -o slurm.%N.%j.out # STDOUT
#SBATCH -e slurm.%N.%j.err # STDERR
#SBATCH --gres=gpu:1

source activate thesis

PREPROCESS_SCRIPTS="scripts/preprocessing"
CORPUS_DIR="data"
BOW_MODELS_DIR="models/bow"
BOW_CORPUS_FILE="wiki_tfidf.mm"
BOW_CORPUS="$BOW_MODELS_DIR/$BOW_CORPUS_FILE"
WORD_IDS="$BOW_MODELS_DIR/wiki_wordids.txt.bz2"


# 0. Download Wikipedia corpus
# echo "Started downloading Wikipedia corpus"
# wget -O $CORPUS_DIR/enwiki-latest-pages-articles.xml.bz2 https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2 
# echo "Downloaded Wikipedia corpus to $CORPUS_DIR"


# 1. --------------- Vectorize corpus (BOW & TF-IDF) ---------------
mkdir -p $BOW_MODELS_DIR 
echo "Started make_wikicorpus"
python $PREPROCESS_SCRIPTS/gensim-make_wikicorpus.py $CORPUS_DIR/enwiki-latest-pages-articles.xml.bz2 $BOW_MODELS_DIR/wiki 10000
echo "Done make_wikicorpus. Find the files in $BOW_MODELS_DIR"
# --> Produces files (in BOW_MODELS_DIR): 
#     - wiki_wordids.txt.bz2: mapping between words and their integer ids
#     - wiki_bow.mm: bag-of-words (word counts) representation in Matrix Market format
#     - wiki_bow.mm.index: index for wiki_bow.mm
#     - wiki_bow.mm.metadata.cpickle: titles of documents
#     - wiki_tfidf.mm: TF-IDF representation in Matrix Market format
#     - wiki_tfidf.mm.index: index for wiki_tfidf.mm
#     - wiki.tfidf_model: TF-IDF model


# 2. --------------- Build LSA model ---------------
LSA_MODEL_DIR="models/lsa"
mkdir -p $LSA_MODEL_DIR
LSA_MODEL="$LSA_MODEL_DIR/wiki_lsi_model.model"

echo "Started building LSA model"
python $PREPROCESS_SCRIPTS/lsa.py $WORD_IDS $BOW_CORPUS $LSA_MODEL
echo "Created LSA model, find it in $LSA_MODEL"


# 3. --------------- Build vocabulary ---------------
VOCAB_FILE="data/vocab.pkl"
SAMPLE_WORDS_FILE="data/200_words.txt" 
CELEX="data/celex_dict.pkl"
CONTROLLED_VOCAB_FILE="data/vocab_controlled.pkl"
CONTROLLED_SAMPLE_WORDS="data/200_words_controlled.txt" 

# Build vocabulary with the 5000 most frequent word types in the corpus and select a sample of 200 of them
echo "Started creating vocabulary"
python $PREPROCESS_SCRIPTS/build_vocab.py $WORD_IDS $BOW_CORPUS $VOCAB_FILE $SAMPLE_WORDS_FILE $CELEX $CONTROLLED_VOCAB_FILE $CONTROLLED_SAMPLE_WORDS
echo "Finished creating vocabularies"