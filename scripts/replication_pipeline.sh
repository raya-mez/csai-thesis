#!/bin/bash
#SBATCH -p GPU # partition (queue)
#SBATCH -N 1 # number of nodes
#SBATCH -t 0-36:00 # time (D-HH:MM)
#SBATCH -o slurm.%N.%j.out # STDOUT
#SBATCH -e slurm.%N.%j.err # STDERR
#SBATCH --gres=gpu:1

source activate thesis

# 0. Download wki corpus
# wget https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2 

# # 1. --------------- Preprocess ---------------
# echo "Started make_wikicorpus"
# python -m gensim.scripts.make_wikicorpus data/enwiki-latest-pages-articles.xml.bz2 data/wiki 10000
# echo "Done make_wikicorpus"
# #     --> produces files (in data/): 
# #         - wiki_wordids.txt.bz2: mapping between words and their integer ids
# #         - wiki_bow.mm: bag-of-words (word counts) representation in Matrix Market format
# #         - wiki_bow.mm.index: index for wiki_bow.mm
# #         - wiki_bow.mm.metadata.cpickle: titles of documents
# #         - wiki_tfidf.mm: TF-IDF representation in Matrix Market format
# #         - wiki_tfidf.mm.index: index for wiki_tfidf.mm
# #         - wiki.tfidf_model: TF-IDF model


# # 2. --------------- Inspect words ---------------
#     # 2.1. --------------- Extract a sample of 200 words to inspect ---------------
# WORDIDS="data/wiki_wordids.txt.bz2"
# BOW_CORPUS="data/wiki_bow.mm"
# OUTPUT_FILE_PATH="data/selected_wordtypes.txt"

# echo "Started extracting 200 word types"
# python scripts/get_200_word_types.py $WORDIDS $BOW_CORPUS $OUTPUT_FILE_PATH
# echo "Finished extracting 200 word types. Find them in $OUTPUT_FILE_PATH"

#     # 2.2. --------------- Inspect if the words are legit ---------------
# INPUT_FILE_PATH="data/selected_wordtypes.txt"
# OUTPUT_FILE_PATH="results/ratio_invalid_en_word_wiki.txt"

# echo "Started inspecting words in data/selected_wordtypes.txt"
# python scripts/inspect_words.py $INPUT_FILE_PATH $OUTPUT_FILE_PATH
# echo "Finished inspecting words in data/selected_wordtypes.txt. Find the ratio of legit words in $OUTPUT_FILE_PATH"


# # 3. --------------- Run LSA model ---------------
# WORDIDS="data/wiki_wordids.txt.bz2"
# TFIDF_MODEL='data/wiki_tfidf.mm'
# LSA_MODEL='models/wiki_lsi_model.model'

# echo "Started training LSA model"
# python scripts/lsa.py $WORDIDS $TFIDF_MODEL $LSA_MODEL
# echo "Finished trainign LSA model"


# # 4. --------------- Build vocabulary ---------------
# WORDIDS="data/wiki_wordids.txt.bz2"
# BOW_CORPUS="data/wiki_bow.mm"
# OUTPUT_FILE="data/vocab.pkl"

# # Build vocabulary containing the 5000 most frequent word types in the corpus
# echo "Started creating vocabulary"
# python scripts/build_vocab.py $WORDIDS $BOW_CORPUS $OUTPUT_FILE
# echo "Finished creating vocabulary. Find it in $OUTPUT_FILE"


# 5. --------------- Cosine distances ---------------
# VOCAB="data/vocab.pkl"
# LSA_MODEL="models/wiki_lsi_model.model"
# # mkdir results/cos_dist_per_word_length
# OUTPUT_FOLDER="results/cos_dist_per_word_length"

# # Compute cosine distance scores for words of the same length (between 3 and 7) in the vocabulary
# echo "Started computing cosine distances per word length"
# python scripts/cos_dist_wordlength.py $VOCAB $LSA_MODEL $OUTPUT_FOLDER
# echo "Finished computing cosine distances per word length. Find the results in $OUTPUT_FOLDER"


# 6. --------------- Edit distances ---------------
# Directory containing files with cosine similarity scores
# DIRECTORY_IN="results/cos_dist_per_word_length"
# # Directory to store aggregated cosine distance and edit distance scores for each word length
# # mkdir results/cos_edit_dist
# DIRECTORY_OUT="results/cos_edit_dist"

# echo "Started computing edit distances"
# python scripts/edit_dist.py $DIRECTORY_IN $DIRECTORY_OUT
# echo "Finished computing edit distances"

# 7. --------------- Pearson correlations ---------------
# Directory containing the CSV files with distance scores
# SIM_SCORES_DIRECTORY="results/cos_edit_dist"
# CORR_SCORES_FILE="results/correlations/pearson_correlations.csv"
# # Create a directory to store correlation plots
# # mkdir results/correlations/corr_plots
# PLOTS_DIRECTORY="results/correlations/corr_plots"

# #  Compute Pearson correlations between cosine distance and edit distance scores for each word length
# echo "Started calculating Pearson correlations"
# python scripts/correlations_per_length.py $SIM_SCORES_DIRECTORY $CORR_SCORES_FILE $PLOTS_DIRECTORY
# echo "Done calculating and plotting correlations"

# # 8. --------------- Random baseline ---------------
# VOCAB="data/vocab.pkl"
# LSA_MODEL="models/wiki_lsi_model.model"
# OUTPUT_FOLDER="results/random_baseline/cos_edit_dist_2.0"

# # Compute cosine and edit distance scores for words of the same length (between 3 and 7) in the vocabulary WITH SHUFFLED VECTORS
# echo "Started computing random baseline cosine and edit distances per word length"
# python scripts/distance_computations/random_baseline_dist_scores.py $VOCAB $LSA_MODEL $OUTPUT_FOLDER
# echo "Finished computing random baseline cosine and edit distances per word length. Find the results in $OUTPUT_FOLDER"

# # 9. --------------- Baseline Pearson correlations ---------------
# # Directory containing the CSV files with distance scores
# SIM_SCORES_DIRECTORY="results/random_baseline/cos_edit_dist"
# CORR_SCORES_FILE="results/random_baseline/pearson_correlations.csv"
# # Create a directory to store correlation plots
# # mkdir results/random_baseline/corr_plots
# PLOTS_DIRECTORY="results/random_baseline/corr_plots"

# #  Compute Pearson correlations between cosine distance and edit distance scores for each word length
# echo "Started calculating Pearson correlations"
# python scripts/correlations_per_length.py $SIM_SCORES_DIRECTORY $CORR_SCORES_FILE $PLOTS_DIRECTORY
# echo "Done calculating and plotting correlations. Find them in $CORR_SCORES_FILE and $PLOTS_DIRECTORY respectively."

# # 10. --------------- Compare correlation ---------------
# REAL_CORR_SCORES="results/correlations/pearson_correlations.csv"
# BASELINE_CORR_SCORES="results/random_baseline/pearson_correlations.csv"
# OUTPUT_FILE="results/correlations/effect_size_random_baseline.csv"

# echo "Started comparing correlations"
# python scripts/fisher_z.py $REAL_CORR_SCORES $BASELINE_CORR_SCORES $OUTPUT_FILE
# echo "Done comparing correlations. Find results in $OUTPUT_FILE"

# 11. --------------- New baseline ---------------
# VOCAB="data/vocab.pkl"
# LSA_MODEL="models/wiki_lsi_model.model"
# mkdir results/new_baseline
# mkdir results/new_baseline/cos_edit_dist
# OUTPUT_FOLDER="results/new_baseline/cos_edit_dist"

# # Compute cosine and edit distance scores for words of the same length (between 3 and 7) in the vocabulary WITH SHUFFLED VECTORS
# echo "Started computing new baseline cosine and edit distances per word length"
# python scripts/new_baseline_dist_scores.py $VOCAB $LSA_MODEL $OUTPUT_FOLDER
# echo "Finished computing new baseline cosine and edit distances per word length. Find the results in $OUTPUT_FOLDER"


# 12. --------------- Baseline Pearson correlations ---------------
# Directory containing the CSV files with distance scores
# SIM_SCORES_DIRECTORY="results/new_baseline/cos_edit_dist"
# CORR_SCORES_FILE="results/new_baseline/pearson_correlations.csv"
# # Create a directory to store correlation plots
# mkdir results/new_baseline/corr_plots
# PLOTS_DIRECTORY="results/new_baseline/corr_plots"

#  Compute Pearson correlations between cosine distance and edit distance scores for each word length
# echo "Started calculating Pearson correlations"
# python scripts/correlations_per_length.py $SIM_SCORES_DIRECTORY $CORR_SCORES_FILE $PLOTS_DIRECTORY
# echo "Done calculating and plotting correlations. Find them in $CORR_SCORES_FILE and $PLOTS_DIRECTORY respectively."


# # 13. --------------- Compare correlation ---------------
# REAL_CORR_SCORES="results/correlations/pearson_correlations.csv"
# BASELINE_CORR_SCORES="results/new_baseline/pearson_correlations.csv"
# OUTPUT_FILE="results/correlations/effect_size_new_baseline.csv"

# echo "Started comparing correlations"
# python scripts/fisher_z.py $REAL_CORR_SCORES $BASELINE_CORR_SCORES $OUTPUT_FILE
# echo "Done comparing correlations. Find results in $OUTPUT_FILE"