#!/bin/bash
#SBATCH -p GPU # partition (queue)
#SBATCH -N 1 # number of nodes
#SBATCH -t 0-36:00 # time (D-HH:MM)
#SBATCH -o slurm.%N.%j.out # STDOUT
#SBATCH -e slurm.%N.%j.err # STDERR
#SBATCH --gres=gpu:1

source activate thesis

INPUT_FILE_PATH="data/selected_wordtypes.txt"
OUTPUT_FILE_PATH="results/ratio_invalid_en_words_wiki.txt"

echo "Started inspecting words in data/selected_wordtypes.txt"
python scripts/inspect_words.py $INPUT_FILE_PATH $OUTPUT_FILE_PATH
echo "Finished inspecting words in data/selected_wordtypes.txt. Find the ratio of legit words in results/ratio_invalid_en_word_wiki.txt"